#!/bin/bash
echo "🎬 Setting up Video Captioner..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Install it from https://python.org"
    exit 1
fi

echo "Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "Run the app with: python3 app.py"
echo "Then open http://localhost:5555"
