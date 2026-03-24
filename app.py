import os
import re
import csv
import json
import glob
import ssl
import subprocess
import tempfile

# Fix SSL certificate issues on some macOS installs
ssl._create_default_https_context = ssl._create_unverified_context
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_file
import anthropic

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Claude caption generation
# ---------------------------------------------------------------------------

def generate_caption(transcript: str, inspiration: str, client_rules: str, example_captions: str, api_key: str) -> str:
    """Generate a caption using Claude based on transcript + style guidance."""
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are a social media caption writer. Your job is to write a single caption for a short-form video (Instagram Reel, TikTok, YouTube Short).

Rules:
- Write ONLY the caption text, nothing else
- No quotation marks around the caption
- Match the tone, length, and style of the example captions exactly
- Use the video transcript to understand what the video is about
- The caption should complement the video, not just repeat the transcript
- Keep it punchy and engaging"""

    user_prompt = f"""Write a caption for this video.

VIDEO TRANSCRIPT:
{transcript}

"""
    if client_rules.strip():
        user_prompt += f"""CLIENT RULES (must follow):
{client_rules}

"""
    if example_captions.strip():
        user_prompt += f"""EXAMPLE CAPTIONS (match this style/tone/length):
{example_captions}

"""
    if inspiration.strip():
        user_prompt += f"""ADDITIONAL STYLE NOTES:
{inspiration}

"""
    user_prompt += "Write the caption now:"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )
    return message.content[0].text.strip()

# ---------------------------------------------------------------------------
# Google Drive public folder scraper
# ---------------------------------------------------------------------------

def scrape_drive_folder(folder_url: str) -> list[dict]:
    """Scrape file names and direct links from a *public* Google Drive folder.

    Uses multiple strategies:
    1. Google Drive API with embedded key (works for truly public folders)
    2. Scrape the HTML page for embedded JSON data
    3. Parse any file IDs and names from the raw page source
    """
    # Extract folder ID from various URL formats
    match = re.search(r'folders/([a-zA-Z0-9_-]+)', folder_url)
    if not match:
        match = re.search(r'id=([a-zA-Z0-9_-]+)', folder_url)
    if not match:
        # Maybe they pasted just the ID
        if re.match(r'^[a-zA-Z0-9_-]{10,}$', folder_url.strip()):
            folder_id = folder_url.strip()
        else:
            raise ValueError("Could not extract folder ID from URL. Make sure the folder is set to 'Anyone with the link'.")
    else:
        folder_id = match.group(1)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".MP4", ".MOV", ".mts", ".MTS"}

    # Strategy 1: Try the public embedLink/export approach
    files = _scrape_drive_embed(folder_id, video_extensions)
    if files:
        return files

    # Strategy 2: Try HTML scrape with multiple patterns
    files = _scrape_drive_html(folder_id, video_extensions)
    if files:
        return files

    raise ValueError(
        f"Could not read files from this Drive folder. "
        f"Make sure the folder sharing is set to 'Anyone with the link' can view. "
        f"Folder ID detected: {folder_id}"
    )


def _scrape_drive_embed(folder_id: str, video_extensions: set) -> list[dict]:
    """Use Google's public embed page which lists files without needing JS rendering."""
    try:
        url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return []

        files = []
        html = resp.text

        # The embedded view contains file entries with IDs and names
        # Pattern: data-id="FILE_ID" ... filename text
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Look for flip entries (embedded folder view format)
        for entry in soup.find_all("div", class_="flip-entry"):
            entry_id = entry.get("id", "")
            file_id = entry_id.replace("entry-", "") if entry_id.startswith("entry-") else entry_id
            if not file_id:
                continue
            title_el = entry.find("div", class_="flip-entry-title")
            if title_el:
                name = title_el.get_text(strip=True)
                if any(name.lower().endswith(ext.lower()) for ext in video_extensions):
                    files.append({
                        "name": name,
                        "drive_link": f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
                        "id": file_id,
                    })

        # Also try parsing raw JS data in the page
        if not files:
            # Google sometimes embeds file data as JSON-like structures
            id_pattern = re.compile(r'\\x22([a-zA-Z0-9_-]{20,60})\\x22')
            name_pattern = re.compile(r'\\x22([^\\]+\.(?:mp4|mov|avi|mkv|webm|m4v))\\x22', re.IGNORECASE)

            file_ids = id_pattern.findall(html)
            file_names = name_pattern.findall(html)

            # Try to pair them up
            seen = set()
            for name in file_names:
                if name not in seen:
                    seen.add(name)
                    # Find the closest preceding file ID
                    name_pos = html.find(name)
                    best_id = None
                    best_dist = float('inf')
                    for fid in file_ids:
                        fid_pos = html.rfind(fid, 0, name_pos)
                        if fid_pos >= 0 and (name_pos - fid_pos) < best_dist:
                            best_dist = name_pos - fid_pos
                            best_id = fid
                    if best_id:
                        files.append({
                            "name": name,
                            "drive_link": f"https://drive.google.com/file/d/{best_id}/view?usp=sharing",
                            "id": best_id,
                        })

        return sorted(files, key=lambda x: x["name"])
    except Exception:
        return []


def _scrape_drive_html(folder_id: str, video_extensions: set) -> list[dict]:
    """Fallback: scrape the regular Drive folder page."""
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return []

        html = resp.text
        files = []

        # Multiple regex patterns to catch Google's various data formats
        patterns = [
            # Pattern 1: ["FILE_ID",["filename.mp4"
            re.compile(r'\["([a-zA-Z0-9_-]{20,60})",\s*\["([^"]+\.(?:mp4|mov|avi|mkv|webm|m4v))"', re.IGNORECASE),
            # Pattern 2: "FILE_ID" ... "filename.mp4" in close proximity
            re.compile(r'"([a-zA-Z0-9_-]{25,45})"[^"]{0,200}"([^"]+\.(?:mp4|mov|avi|mkv|webm|m4v))"', re.IGNORECASE),
            # Pattern 3: data encoded with \x22 escapes
            re.compile(r'\\x22([a-zA-Z0-9_-]{25,45})\\x22[^\\]{0,200}\\x22([^\\]+\.(?:mp4|mov|avi|mkv|webm|m4v))\\x22', re.IGNORECASE),
        ]

        seen_names = set()
        for pattern in patterns:
            for m in pattern.finditer(html):
                file_id = m.group(1)
                file_name = m.group(2)
                if file_name not in seen_names:
                    seen_names.add(file_name)
                    files.append({
                        "name": file_name,
                        "drive_link": f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
                        "id": file_id,
                    })

        return sorted(files, key=lambda x: x["name"])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def get_ffmpeg_path() -> str:
    """Get ffmpeg path — prefer system install, fall back to imageio_ffmpeg."""
    try:
        result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    return "ffmpeg"


def transcribe_video(video_path: str, model_name: str = "base") -> str:
    """Transcribe a video file using Whisper."""
    import whisper

    ffmpeg_path = get_ffmpeg_path()
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + ":" + os.environ.get("PATH", "")

    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)
    return result["text"].strip()


# Store state in memory for the session
SESSION = {
    "drive_files": [],
    "local_files": [],
    "matched": [],
    "transcriptions": {},
    "captions": {},
    "config": {},
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/pick-folder", methods=["POST"])
def pick_folder():
    """Open a native folder picker dialog using AppleScript (macOS)."""
    try:
        result = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select Video Folder")'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            folder = result.stdout.strip().rstrip("/")
            return jsonify({"success": True, "path": folder})
        return jsonify({"success": False, "error": "No folder selected"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/scan-drive", methods=["POST"])
def scan_drive():
    data = request.json
    folder_url = data.get("folder_url", "")
    try:
        files = scrape_drive_folder(folder_url)
        if not files:
            return jsonify({
                "success": False,
                "error": "No video files found in this Drive folder. Make sure: (1) the folder is set to 'Anyone with the link' can view, and (2) it contains video files (mp4, mov, etc.)"
            }), 400
        SESSION["drive_files"] = files
        return jsonify({"success": True, "files": files, "count": len(files)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/scan-local", methods=["POST"])
def scan_local():
    data = request.json
    folder_path = data.get("folder_path", "")

    if not os.path.isdir(folder_path):
        return jsonify({"success": False, "error": "Folder not found"}), 400

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".MP4", ".MOV"}
    files = []
    for f in sorted(os.listdir(folder_path)):
        if Path(f).suffix in video_extensions:
            files.append({
                "name": f,
                "path": os.path.join(folder_path, f),
            })

    SESSION["local_files"] = files
    return jsonify({"success": True, "files": files})


@app.route("/api/match-files", methods=["POST"])
def match_files():
    """Match local files to Drive files by filename."""
    drive_files = SESSION.get("drive_files", [])
    local_files = SESSION.get("local_files", [])

    # Build lookup by normalized name (lowercase, no extension)
    def normalize(name):
        return Path(name).stem.lower().strip()

    drive_lookup = {}
    for df in drive_files:
        drive_lookup[normalize(df["name"])] = df

    matched = []
    unmatched_local = []
    for lf in local_files:
        key = normalize(lf["name"])
        if key in drive_lookup:
            matched.append({
                "local_name": lf["name"],
                "local_path": lf["path"],
                "drive_name": drive_lookup[key]["name"],
                "drive_link": drive_lookup[key]["drive_link"],
            })
        else:
            unmatched_local.append(lf["name"])

    unmatched_drive = [df["name"] for df in drive_files if normalize(df["name"]) not in {normalize(lf["name"]) for lf in local_files}]

    SESSION["matched"] = matched
    return jsonify({
        "success": True,
        "matched": matched,
        "unmatched_local": unmatched_local,
        "unmatched_drive": unmatched_drive,
    })


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Transcribe a single video."""
    data = request.json
    video_path = data.get("video_path", "")
    model_name = data.get("model", "base")

    if not os.path.isfile(video_path):
        return jsonify({"success": False, "error": "File not found"}), 400

    try:
        text = transcribe_video(video_path, model_name)
        SESSION["transcriptions"][video_path] = text
        return jsonify({"success": True, "transcript": text})
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Transcription error for {video_path}: {error_detail}")
        return jsonify({"success": False, "error": f"{type(e).__name__}: {str(e)}"}), 500


@app.route("/api/transcribe-all", methods=["POST"])
def transcribe_all():
    """Transcribe all matched videos with streaming progress."""
    data = request.json
    model_name = data.get("model", "base")
    matched = SESSION.get("matched", [])

    def generate():
        total = len(matched)
        for i, m in enumerate(matched):
            path = m["local_path"]
            name = m["local_name"]
            try:
                # Send progress update
                yield f"data: {json.dumps({'type': 'progress', 'current': i+1, 'total': total, 'name': name, 'status': 'transcribing'})}\n\n"
                text = transcribe_video(path, model_name)
                SESSION["transcriptions"][path] = text
                yield f"data: {json.dumps({'type': 'result', 'current': i+1, 'total': total, 'name': name, 'transcript': text, 'success': True})}\n\n"
            except Exception as e:
                error_msg = str(e)
                yield f"data: {json.dumps({'type': 'result', 'current': i+1, 'total': total, 'name': name, 'error': error_msg, 'success': False})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'total': total})}\n\n"

    return app.response_class(generate(), mimetype='text/event-stream')


@app.route("/api/generate-caption", methods=["POST"])
def generate_caption_route():
    """Generate a caption for a single video using Claude."""
    data = request.json
    transcript = data.get("transcript", "")
    inspiration = data.get("inspiration", "")
    client_rules = data.get("client_rules", "")
    example_captions = data.get("example_captions", "")
    api_key = data.get("api_key", "")

    if not api_key:
        return jsonify({"success": False, "error": "Anthropic API key required"}), 400
    if not transcript:
        return jsonify({"success": False, "error": "No transcript provided"}), 400

    try:
        caption = generate_caption(transcript, inspiration, client_rules, example_captions, api_key)
        return jsonify({"success": True, "caption": caption})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate-all-captions", methods=["POST"])
def generate_all_captions_route():
    """Generate captions for all transcribed videos."""
    data = request.json
    transcripts = data.get("transcripts", {})
    inspiration = data.get("inspiration", "")
    client_rules = data.get("client_rules", "")
    example_captions = data.get("example_captions", "")
    api_key = data.get("api_key", "")

    if not api_key:
        return jsonify({"success": False, "error": "Anthropic API key required"}), 400

    results = {}
    for video_path, transcript in transcripts.items():
        try:
            caption = generate_caption(transcript, inspiration, client_rules, example_captions, api_key)
            results[video_path] = {"caption": caption, "success": True}
        except Exception as e:
            results[video_path] = {"caption": "", "error": str(e), "success": False}

    return jsonify({"success": True, "results": results})


@app.route("/api/save-captions", methods=["POST"])
def save_captions():
    """Save edited captions from the UI."""
    data = request.json
    captions = data.get("captions", {})
    SESSION["captions"] = captions
    return jsonify({"success": True})


@app.route("/api/export-csv", methods=["POST"])
def export_csv():
    """Generate Metricool CSV."""
    data = request.json
    config = data.get("config", {})
    captions = data.get("captions", [])

    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    posts_per_day = int(config["posts_per_day"])
    times = config["times"]  # list of "HH:MM:SS" strings
    platforms = config.get("platforms", {})

    # Metricool CSV header
    header = [
        "Text", "Date", "Time", "Draft",
        "Facebook", "Twitter/X", "LinkedIn", "GBP", "Instagram", "Pinterest",
        "TikTok", "Youtube", "Threads", "Bluesky",
        "Picture Url 1", "Picture Url 2", "Picture Url 3", "Picture Url 4",
        "Picture Url 5", "Picture Url 6", "Picture Url 7", "Picture Url 8",
        "Picture Url 9", "Picture Url 10",
        "Alt text picture 1", "Alt text picture 2", "Alt text picture 3",
        "Alt text picture 4", "Alt text picture 5", "Alt text picture 6",
        "Alt text picture 7", "Alt text picture 8", "Alt text picture 9",
        "Alt text picture 10",
        "Document title", "Shortener", "Video Thumbnail Url", "Video Cover Frame",
        "Twitter/X Can reply", "Twitter/X Type",
        "Twitter/X Poll Duration minutes", "Twitter/X Poll Option 1",
        "Twitter/X Poll Option 2", "Twitter/X Poll Option 3", "Twitter/X Poll Option 4",
        "Pinterest Board", "Pinterest Pin Title", "Pinterest Pin Link",
        "Pinterest Pin New Format",
        "Instagram Post Type", "Instagram Show Reel On Feed",
        "Youtube Video Title", "Youtube Video Type", "Youtube Video Privacy",
        "Youtube video for kids", "Youtube Video Category", "Youtube Video Tags",
        "Youtube playlist",
        "GBP Post Type",
        "Facebook Post Type", "Facebook Title",
        "First Comment Text",
        "TikTok Title", "TikTok disable comments", "TikTok disable duet",
        "TikTok disable stitch", "TikTok Post Privacy",
        "TikTok Branded Content", "TikTok Your Brand", "TikTok Auto Add Music",
        "TikTok Photo Cover Index",
        "TikTok musicId", "TikTok music title", "TikTok music author",
        "TikTok music previewUrl", "TikTok music thumbnailUrl",
        "TikTok music soundVolume", "TikTok music originalVolume",
        "TikTok music startMillis", "TikTok music endMillis",
        "TikTok Ai generated content",
        "LinkedIn Type", "LinkedIn Poll Question",
        "LinkedIn Poll Option 1", "LinkedIn Poll Option 2",
        "LinkedIn Poll Option 3", "LinkedIn Poll Option 4",
        "LinkedIn Poll Duration", "LinkedIn Show link preview",
        "LinkedIn Images as Carousel",
        "Threads Reply Control", "Threads Is Spoiler", "Threads Post Type",
        "Brand name",
    ]

    rows = []
    current_date = start_date
    time_index = 0

    for caption_data in captions:
        caption_text = caption_data.get("caption", "")
        drive_link = caption_data.get("drive_link", "")

        post_time = times[time_index % len(times)]

        row = {col: "" for col in header}
        row["Text"] = caption_text
        row["Date"] = current_date.strftime("%Y-%m-%d")
        row["Time"] = post_time
        row["Draft"] = "FALSE"

        # Platforms
        row["Facebook"] = "TRUE" if platforms.get("facebook") else "FALSE"
        row["Twitter/X"] = "FALSE"
        row["LinkedIn"] = "FALSE"
        row["GBP"] = "FALSE"
        row["Instagram"] = "TRUE" if platforms.get("instagram") else "FALSE"
        row["Pinterest"] = "FALSE"
        row["TikTok"] = "TRUE" if platforms.get("tiktok") else "FALSE"
        row["Youtube"] = "TRUE" if platforms.get("youtube") else "FALSE"
        row["Threads"] = "TRUE" if platforms.get("threads") else "FALSE"
        row["Bluesky"] = "FALSE"

        # Video link
        row["Picture Url 1"] = drive_link

        # Platform-specific settings for video
        if platforms.get("instagram"):
            row["Instagram Post Type"] = "REEL"
            row["Instagram Show Reel On Feed"] = "true"
        if platforms.get("youtube"):
            row["Youtube Video Title"] = caption_text[:100] if caption_text else ""
            row["Youtube Video Type"] = "SHORT"
            row["Youtube Video Privacy"] = "PUBLIC"
            row["Youtube video for kids"] = "FALSE"
        if platforms.get("facebook"):
            row["Facebook Post Type"] = "REEL"
        if platforms.get("tiktok"):
            row["TikTok Post Privacy"] = "PUBLIC_TO_EVERYONE"
            row["TikTok disable comments"] = "FALSE"
            row["TikTok disable duet"] = "FALSE"
            row["TikTok disable stitch"] = "FALSE"

        rows.append(row)

        time_index += 1
        if time_index % posts_per_day == 0 or time_index >= len(times):
            if time_index >= posts_per_day:
                current_date += timedelta(days=1)
                time_index = 0

    # Write CSV
    output_path = os.path.join(tempfile.gettempdir(), "metricool_export.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return send_file(output_path, as_attachment=True, download_name="metricool_export.csv")


if __name__ == "__main__":
    print("\n✅ Video Captioner is running!")
    print("   Open http://localhost:5555 in your browser\n")
    app.run(debug=True, port=5555, use_reloader=False)
