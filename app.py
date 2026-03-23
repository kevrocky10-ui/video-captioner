import os
import re
import csv
import json
import glob
import subprocess
import tempfile
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Google Drive public folder scraper
# ---------------------------------------------------------------------------

def scrape_drive_folder(folder_url: str) -> list[dict]:
    """Scrape file names and direct links from a *public* Google Drive folder."""
    # Extract folder ID from URL
    match = re.search(r'folders/([a-zA-Z0-9_-]+)', folder_url)
    if not match:
        raise ValueError("Could not extract folder ID from URL")
    folder_id = match.group(1)

    files = []
    page_token = None

    while True:
        api_url = (
            f"https://www.googleapis.com/drive/v3/files"
            f"?q='{folder_id}'+in+parents"
            f"&fields=nextPageToken,files(id,name,mimeType,webViewLink)"
            f"&pageSize=1000"
            f"&key=AIzaSyC1qbk72OEhAETMjQ9CkEN-r-0n4B3FMHE"
        )
        if page_token:
            api_url += f"&pageToken={page_token}"

        resp = requests.get(api_url, timeout=30)

        # Fallback: try scraping HTML if API key doesn't work
        if resp.status_code != 200:
            return _scrape_drive_html(folder_id)

        data = resp.json()
        for f in data.get("files", []):
            if f.get("mimeType", "").startswith("video/"):
                files.append({
                    "name": f["name"],
                    "drive_link": f.get("webViewLink", f"https://drive.google.com/file/d/{f['id']}/view"),
                    "id": f["id"],
                })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return sorted(files, key=lambda x: x["name"])


def _scrape_drive_html(folder_id: str) -> list[dict]:
    """Fallback HTML scraper for public Drive folders."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    files = []
    # Google Drive embeds file data in JS — look for patterns
    pattern = re.compile(r'\["([a-zA-Z0-9_-]{20,})",\["([^"]+\.(mp4|mov|avi|mkv|webm|m4v))"', re.IGNORECASE)
    for match in pattern.finditer(resp.text):
        file_id = match.group(1)
        file_name = match.group(2)
        files.append({
            "name": file_name,
            "drive_link": f"https://drive.google.com/file/d/{file_id}/view",
            "id": file_id,
        })

    return sorted(files, key=lambda x: x["name"])


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


@app.route("/api/scan-drive", methods=["POST"])
def scan_drive():
    data = request.json
    folder_url = data.get("folder_url", "")
    try:
        files = scrape_drive_folder(folder_url)
        SESSION["drive_files"] = files
        return jsonify({"success": True, "files": files})
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
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/transcribe-all", methods=["POST"])
def transcribe_all():
    """Transcribe all matched videos."""
    data = request.json
    model_name = data.get("model", "base")
    matched = SESSION.get("matched", [])

    results = []
    for i, m in enumerate(matched):
        path = m["local_path"]
        try:
            text = transcribe_video(path, model_name)
            SESSION["transcriptions"][path] = text
            results.append({"name": m["local_name"], "transcript": text, "success": True})
        except Exception as e:
            results.append({"name": m["local_name"], "error": str(e), "success": False})

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
    app.run(debug=True, port=5555)
