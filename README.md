# SafetyEye - Streamlit Application

Simple, user-friendly AI-powered PPE detection system.

## Features

- 📸 **Image Detection**: Upload and detect PPE violations in images
- 🎥 **Video Processing**: Process videos with progress tracking
- 📊 **Dashboard**: View violation statistics and recent incidents

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

Or double-click `run_streamlit.bat` on Windows.

The app will open at `http://localhost:8503` (default port)

## Usage

### Image Detection
1. Go to "📸 Image Detection" tab
2. Upload an image (PNG, JPG, JPEG)
3. Adjust confidence and IoU thresholds if needed
4. Click "🔍 Detect Violations"
5. View results and download annotated image

### Video Processing
1. Go to "🎥 Video Processing" tab
2. Upload a video (MP4, AVI, MOV, MKV)
3. Adjust detection settings
4. Click "🎬 Process Video"
5. Wait for processing (progress bar shows status)
6. Download the annotated video

### Dashboard
- View total violations and statistics
- See violation distribution chart
- Review recent incidents table

## Configuration

Create a `.env` file (optional):
```
PROJECT_ROOT=D:\Desktop\coding\Flask_app
MODEL_PATH=D:\desktop\coding\Flask_app\model\best.pt
STREAM_CONF=0.25
STREAM_IOU=0.45
```

## Troubleshooting

**Port already in use:**
- Check what's using the port: `netstat -ano | findstr :8503`
- Kill the process or change port in `.streamlit/config.toml`
- Or specify a different port: `streamlit run app.py --server.port 8504`

**Model not loading:**
- Check model path in `.env` or default location
- Verify model file exists

**PyTorch warnings:**
- These are harmless and can be ignored
- The app uses polling file watcher to minimize the
## File Structure

```
Flask_app/
├── app.py              # Main application (single file)
├── utils.py            # Shared utilities
├── requirements.txt    # Dependencies
└── .streamlit/
    └── config.toml     # Streamlit configuration
```

## Notes

- All violations are logged to `logs/violations.sqlite`
- Processed files saved to `flask_outputs/`
- Snapshots saved to `logs/snaps/`
- Email alerts sent if SMTP configured in `.env`
