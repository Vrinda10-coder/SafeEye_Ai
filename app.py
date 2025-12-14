# ===========================
# SafetyEye AI - Production PPE Detection System
# Main Application Entry Point
# ===========================
import streamlit as st # type: ignore
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv # type: ignore

# Fix for Streamlit file watcher issue with PyTorch
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'poll'
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*RuntimeError.*torch.*")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SafetyEye AI - PPE Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #3B82F6;
    }
    .sub-header {
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .upload-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed rgba(59, 130, 246, 0.3);
        margin: 1rem 0;
    }
    .result-section {
        background: rgba(26, 31, 46, 0.6);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    .violation-card {
        background: rgba(239, 68, 68, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
        margin: 0.5rem 0;
    }
    .success-card {
        background: rgba(16, 185, 129, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 0.5rem 0;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Import utilities
from utils import (
    BASE_DIR, MODEL_PATH, OUTPUT_DIR, SNAP_DIR,
    init_db, load_model, read_violations_df,
    annotate_frame_with_model, evaluate_frame_rules,
    save_violation_record, send_email_alert, now_str,
    allowed_file, validate_file_size, MAX_IMAGE_SIZE, MAX_VIDEO_SIZE,
    ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT
)

# Initialize database
init_db()

# Load model (cached - loaded once at startup)
@st.cache_resource
def get_model():
    """Load YOLO model once at application startup."""
    return load_model()

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# Load model
if st.session_state.model is None:
    with st.spinner("🔄 Loading YOLO model..."):
        st.session_state.model = get_model()
        if st.session_state.model is None:
            st.error("❌ **Model not loaded.** Please check model path in configuration.")
            st.stop()
        else:
            st.success("✅ **Model loaded successfully!**")

# Sidebar Navigation
st.sidebar.title("🛡️ SafetyEye AI")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📸 Upload Image", "🎥 Upload Video", "📹 Live Camera", "📊 Violation Logs", "⚙️ Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** {Path(MODEL_PATH).name if MODEL_PATH else 'Not set'}")
st.sidebar.markdown(f"**Status:** {'🟢 Active' if st.session_state.model else '🔴 Inactive'}")

# ========== HOME PAGE ==========
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🛡️ SafetyEye AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Personal Protective Equipment Detection System</p>', unsafe_allow_html=True)
    
    # Quick stats
    df = read_violations_df(limit=1000)
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    unique_types = df["violation_type"].nunique() if not df.empty else 0
    
    with col1:
        st.metric("Total Violations", total)
    with col2:
        st.metric("Violation Types", unique_types)
    with col3:
        st.metric("System Status", "🟢 Active")
    with col4:
        last_ts = df["ts"].iloc[0] if not df.empty else "Never"
        st.metric("Last Detection", last_ts[:10] if len(str(last_ts)) > 10 else last_ts)
    
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>📸 Image Detection</h3>
            <p>Upload images to detect PPE violations instantly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>🎥 Video Processing</h3>
            <p>Process videos frame-by-frame for comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>📹 Live Monitoring</h3>
            <p>Real-time detection with instant email alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👆 **Use the sidebar to navigate to different sections**")

# ========== IMAGE UPLOAD PAGE ==========
elif page == "📸 Upload Image":
    st.markdown('<h1 class="main-header">📸 Image Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to detect PPE violations</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=list(ALLOWED_IMAGE_EXT),
        help=f"Supported formats: {', '.join(ALLOWED_IMAGE_EXT)}. Maximum size: {MAX_IMAGE_SIZE / (1024*1024):.0f}MB"
    )
    
    # Detection settings
    st.markdown("**Detection Settings:**")
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    with col2:
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Validate file
        file_bytes = uploaded_file.read()
        is_valid, error_msg = validate_file_size(file_bytes, MAX_IMAGE_SIZE)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
        elif not allowed_file(uploaded_file.name, ALLOWED_IMAGE_EXT):
            st.error(f"❌ Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_EXT)}")
        else:
            if st.button("🔍 Detect PPE Violations", type="primary", use_container_width=True):
                st.session_state.processing = True
                
                with st.spinner("🔄 Processing image... Please wait..."):
                    try:
                        import cv2
                        import numpy as np
                        from PIL import Image
                        import tempfile
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name
                        
                        # Load image
                        pil_img = Image.open(tmp_path).convert("RGB")
                        frame = np.array(pil_img)[:, :, ::-1]  # RGB → BGR
                        
                        # Run detection
                        annotated, detections = annotate_frame_with_model(
                            frame, conf=conf_threshold, iou=iou_threshold
                        )
                        
                        # Evaluate violations
                        violations = evaluate_frame_rules(detections)
                        violation_count = len(violations)
                        violation_types = list(set([v["type"] for v in violations])) if violations else []
                        
                        # Save annotated image
                        out_name = f"annot_{now_str()}_{uploaded_file.name}"
                        out_path = OUTPUT_DIR / out_name
                        cv2.imwrite(str(out_path), annotated)
                        
                        # Save violations and snapshots
                        snap_path = None
                        if violations:
                            snap_name = f"{now_str()}_{'_'.join(v['type'].replace(' ','_') for v in violations[:3])}.jpg"
                            snap_path = SNAP_DIR / snap_name
                            cv2.imwrite(str(snap_path), annotated)
                            
                            # Log each violation
                            for v in violations:
                                save_violation_record(
                                    violation_type=v["type"],
                                    snap_path=str(snap_path),
                                    person_id=v.get("person_id"),
                                    source="image_upload"
                                )
                        
                        # Store result in session state for result page
                        st.session_state.last_result = {
                            "type": "image",
                            "output_name": out_name,
                            "output_path": str(out_path),
                            "snapshot_path": str(snap_path) if snap_path else None,
                            "violations": violations,
                            "violation_count": violation_count,
                            "violation_types": violation_types,
                            "detections": len(detections),
                            "timestamp": now_str()
                        }
                        
                        os.unlink(tmp_path)
                        st.session_state.processing = False
                        
                        # Redirect to result page
                        st.success("✅ **Processing complete!** Redirecting to results...")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ **Error processing image:** {str(e)}")
                        st.exception(e)
                        st.session_state.processing = False
                        os.unlink(tmp_path) if 'tmp_path' in locals() else None

# ========== VIDEO UPLOAD PAGE ==========
elif page == "🎥 Upload Video":
    st.markdown('<h1 class="main-header">🎥 Video Processing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a video file to detect PPE violations</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=list(ALLOWED_VIDEO_EXT),
        help=f"Supported formats: {', '.join(ALLOWED_VIDEO_EXT)}. Maximum size: {MAX_VIDEO_SIZE / (1024*1024):.0f}MB"
    )
    
    st.markdown("**Detection Settings:**")
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold_v = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="conf_v")
    with col2:
        iou_threshold_v = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05, key="iou_v")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_video is not None:
        file_bytes_v = uploaded_video.read()
        is_valid_v, error_msg_v = validate_file_size(file_bytes_v, MAX_VIDEO_SIZE)
        
        if not is_valid_v:
            st.error(f"❌ {error_msg_v}")
        elif not allowed_file(uploaded_video.name, ALLOWED_VIDEO_EXT):
            st.error(f"❌ Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXT)}")
        else:
            if st.button("🎬 Process Video", type="primary", use_container_width=True):
                st.session_state.processing = True
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    import cv2
                    import tempfile
                    
                    # Save video
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as tmp:
                        tmp.write(file_bytes_v)
                        tmp_path = tmp.name
                    
                    status_text.text("📹 Opening video file...")
                    
                    # Open video
                    cap = cv2.VideoCapture(tmp_path)
                    if not cap.isOpened():
                        st.error("❌ Failed to open video file")
                        os.unlink(tmp_path)
                        st.session_state.processing = False
                        st.stop()
                    
                    fps = cap.get(cv2.CAP_PROP_FPS) or 20
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                    
                    # Create output
                    out_name_v = f"annot_{now_str()}_{uploaded_video.name}"
                    out_path_v = OUTPUT_DIR / out_name_v
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path_v), fourcc, fps, (w, h))
                    
                    frame_count = 0
                    total_violations = 0
                    violation_types_set = set()
                    last_snap = None
                    all_violations = []
                    
                    status_text.text("🔄 Processing frames...")
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        if total_frames > 0:
                            progress = min(frame_count / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_count}/{total_frames}...")
                        
                        # Detect
                        annotated, detections = annotate_frame_with_model(
                            frame, conf=conf_threshold_v, iou=iou_threshold_v
                        )
                        writer.write(annotated)
                        
                        # Check violations
                        violations = evaluate_frame_rules(detections)
                        if violations:
                            total_violations += len(violations)
                            for v in violations:
                                violation_types_set.add(v["type"])
                                all_violations.append(v)
                            
                            # Save snapshot every 30 frames or first violation
                            if frame_count % 30 == 0 or last_snap is None:
                                snap_name = f"{now_str()}_{'_'.join(list(violation_types_set)[:3])}.jpg"
                                snap_path = SNAP_DIR / snap_name
                                cv2.imwrite(str(snap_path), annotated)
                                last_snap = snap_path
                                
                                # Log violations
                                for v in violations:
                                    save_violation_record(
                                        violation_type=v["type"],
                                        snap_path=str(snap_path),
                                        person_id=v.get("person_id"),
                                        source=f"video:{uploaded_video.name}"
                                    )
                    
                    cap.release()
                    writer.release()
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    # Store result
                    st.session_state.last_result = {
                        "type": "video",
                        "output_name": out_name_v,
                        "output_path": str(out_path_v),
                        "snapshot_path": str(last_snap) if last_snap else None,
                        "violations": all_violations,
                        "violation_count": total_violations,
                        "violation_types": list(violation_types_set),
                        "total_frames": frame_count,
                        "timestamp": now_str()
                    }
                    
                    os.unlink(tmp_path)
                    st.session_state.processing = False
                    
                    # Redirect
                    st.success("✅ **Video processing complete!** Redirecting to results...")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ **Error processing video:** {str(e)}")
                    st.exception(e)
                    progress_bar.empty()
                    status_text.empty()
                    st.session_state.processing = False
                    if 'tmp_path' in locals():
                        os.unlink(tmp_path)

# ========== RESULT PAGES (shown after processing) ==========
# Check if we have a result to show
if st.session_state.last_result and not st.session_state.processing:
    result = st.session_state.last_result
    
    if result["type"] == "image":
        st.markdown('<h1 class="main-header">📸 Image Detection Results</h1>', unsafe_allow_html=True)
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        
        # Display annotated image
        import cv2
        annotated_img = cv2.imread(result["output_path"])
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True, caption="Annotated Image with Detections")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", result["detections"])
        with col2:
            st.metric("Violations Found", result["violation_count"])
        with col3:
            st.metric("Violation Types", len(result["violation_types"]))
        
        # Violations
        if result["violations"]:
            st.markdown("### ⚠️ Violations Detected")
            violation_summary = {}
            for v in result["violations"]:
                v_type = v["type"]
                violation_summary[v_type] = violation_summary.get(v_type, 0) + 1
            
            for v_type, count in violation_summary.items():
                st.markdown(f'<div class="violation-card"><strong>{v_type}:</strong> {count} occurrence(s)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card"><strong>✅ No violations detected!</strong> All safety requirements are met.</div>', unsafe_allow_html=True)
        
        # Download
        with open(result["output_path"], "rb") as f:
            st.download_button(
                "💾 Download Annotated Image",
                f.read(),
                result["output_name"],
                "image/jpeg",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear result after showing
        if st.button("🔄 Process Another Image", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()
    
    elif result["type"] == "video":
        st.markdown('<h1 class="main-header">🎥 Video Processing Results</h1>', unsafe_allow_html=True)
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", result["total_frames"])
        with col2:
            st.metric("Violations Found", result["violation_count"])
        with col3:
            st.metric("Violation Types", len(result["violation_types"]))
        
        # Violations
        if result["violation_types"]:
            st.markdown("### ⚠️ Violations Detected")
            for v_type in result["violation_types"]:
                count = sum(1 for v in result["violations"] if v["type"] == v_type)
                st.markdown(f'<div class="violation-card"><strong>{v_type}:</strong> {count} occurrence(s)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card"><strong>✅ No violations detected!</strong> All safety requirements are met.</div>', unsafe_allow_html=True)
        
        # Preview snapshot
        if result["snapshot_path"] and os.path.exists(result["snapshot_path"]):
            import cv2
            preview = cv2.imread(result["snapshot_path"])
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="Preview Frame with Violations", use_container_width=True)
        
        # Video download
        if os.path.exists(result["output_path"]):
            with open(result["output_path"], "rb") as f:
                st.download_button(
                    "💾 Download Processed Video",
                    f.read(),
                    result["output_name"],
                    "video/mp4",
                    use_container_width=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear result
        if st.button("🔄 Process Another Video", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

# ========== LIVE CAMERA PAGE ==========
elif page == "📹 Live Camera":
    st.markdown('<h1 class="main-header">📹 Live Camera Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time PPE violation detection with email alerts</p>', unsafe_allow_html=True)
    
    # Camera settings
    col1, col2 = st.columns(2)
    with col1:
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 Try different indices (0, 1, 2...) if camera doesn't work")
    
    col1, col2 = st.columns(2)
    with col1:
        conf_threshold_live = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05, key="conf_live")
    with col2:
        iou_threshold_live = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05, key="iou_live")
    
    # Stream control
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'last_violation_time' not in st.session_state:
        st.session_state.last_violation_time = {}
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Stream", type="primary", use_container_width=True, disabled=st.session_state.streaming):
            st.session_state.streaming = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Stream", use_container_width=True, disabled=not st.session_state.streaming):
            st.session_state.streaming = False
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except:
                    pass
                st.session_state.cap = None
            st.rerun()
    
    # Stream display
    if st.session_state.streaming:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            import cv2
            import time
            
            # Initialize camera
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(camera_index)
                if not st.session_state.cap.isOpened():
                    st.error(f"❌ Failed to open camera {camera_index}")
                    st.session_state.streaming = False
                    st.stop()
            
            status_placeholder.success("🟢 **Stream Active** - Monitoring for violations...")
            
            # Process frames
            ret, frame = st.session_state.cap.read()
            if ret:
                # Resize if too large
                h, w = frame.shape[:2]
                max_side = 960
                if max(h, w) > max_side:
                    scale = max_side / float(max(h, w))
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                # Run detection
                annotated, detections = annotate_frame_with_model(
                    frame, conf=conf_threshold_live, iou=iou_threshold_live
                )
                
                # Evaluate violations
                violations = evaluate_frame_rules(detections)
                
                # Log violations (with duplicate prevention)
                if violations:
                    current_time = time.time()
                    snap_path = SNAP_DIR / f"{now_str()}_{'_'.join(v['type'].replace(' ','_') for v in violations[:3])}.jpg"
                    cv2.imwrite(str(snap_path), annotated)
                    
                    for v in violations:
                        v_key = f"{v['type']}_{v.get('person_id', 0)}"
                        # Prevent duplicate emails within 5 seconds
                        if v_key not in st.session_state.last_violation_time or \
                           (current_time - st.session_state.last_violation_time[v_key]) > 5:
                            
                            save_violation_record(
                                v["type"],
                                str(snap_path),
                                person_id=v.get("person_id"),
                                source="live_stream"
                            )
                            
                            send_email_alert(
                                subject=f"SafetyEye Alert — {v['type']}",
                                body=f"PPE Violation Detected\n\nType: {v['type']}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nSource: Live Camera Stream",
                                attachment_path=str(snap_path)
                            )
                            
                            st.session_state.last_violation_time[v_key] = current_time
                    
                    status_placeholder.warning(f"⚠️ **{len(violations)} violation(s) detected!** Email alerts sent.")
                
                # Display frame
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, use_container_width=True, channels="RGB")
            
            # Auto-refresh for live stream
            time.sleep(0.1)
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ **Stream error:** {str(e)}")
            st.session_state.streaming = False
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except:
                    pass
                st.session_state.cap = None
    
    else:
        st.info("👆 **Click 'Start Stream' to begin live monitoring**")

# ========== VIOLATION LOGS PAGE ==========
elif page == "📊 Violation Logs":
    st.markdown('<h1 class="main-header">📊 Violation Logs</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Review all detected PPE violations</p>', unsafe_allow_html=True)
    
    # Load violations
    df = read_violations_df(limit=1000)
    
    if df.empty:
        st.info("📭 **No violations recorded yet.**")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        unique_types = df["violation_type"].nunique()
        
        with col1:
            st.metric("Total Violations", total)
        with col2:
            st.metric("Violation Types", unique_types)
        with col3:
            type_counts = df["violation_type"].value_counts()
            most_common = type_counts.index[0] if not type_counts.empty else "N/A"
            st.metric("Most Common", most_common)
        with col4:
            last_ts = df["ts"].iloc[0] if not df.empty else "Never"
            st.metric("Last Detection", last_ts[:16] if len(str(last_ts)) > 16 else last_ts)
        
        st.markdown("---")
        
        # Violation distribution chart
        if not df.empty:
            st.subheader("Violation Distribution")
            st.bar_chart(type_counts)
        
        st.markdown("---")
        
        # Violations table
        st.subheader("Recent Violations")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox("Filter by Type", ["All"] + list(df["violation_type"].unique()) if not df.empty else ["All"])
        with col2:
            filter_source = st.selectbox("Filter by Source", ["All"] + list(df["video_source"].dropna().unique()) if not df.empty else ["All"])
        
        # Apply filters
        display_df = df.copy()
        if filter_type != "All":
            display_df = display_df[display_df["violation_type"] == filter_type]
        if filter_source != "All":
            display_df = display_df[display_df["video_source"] == filter_source]
        
        # Display table
        if not display_df.empty:
            for idx, row in display_df.head(50).iterrows():
                with st.expander(f"🔴 {row['violation_type']} - {row['ts']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Violation Type:** {row['violation_type']}")
                        st.write(f"**Timestamp:** {row['ts']}")
                        st.write(f"**Source:** {row.get('video_source', 'Unknown')}")
                        if row.get('person_id') is not None:
                            st.write(f"**Person ID:** {row['person_id']}")
                    
                    with col2:
                        # Show snapshot if available
                        snap_path = row.get('snapshot_path')
                        if snap_path and os.path.exists(snap_path):
                            import cv2
                            snap_img = cv2.imread(snap_path)
                            snap_rgb = cv2.cvtColor(snap_img, cv2.COLOR_BGR2RGB)
                            st.image(snap_rgb, caption="Violation Snapshot", use_container_width=True)
                        else:
                            st.info("No snapshot available")
        else:
            st.info("No violations match the selected filters.")

# ========== SETTINGS PAGE ==========
elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.subheader("Model Configuration")
    st.info(f"**Model Path:** {MODEL_PATH}")
    st.info(f"**Model Status:** {'✅ Loaded' if st.session_state.model else '❌ Not Loaded'}")
    
    if st.session_state.model and hasattr(st.session_state.model, "names"):
        st.info(f"**Detection Classes:** {len(st.session_state.model.names)} classes available")
    
    st.markdown("---")
    
    st.subheader("Email Alert Configuration")
    st.info("Configure email settings in `.env` file:")
    st.code("""
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_password
ALERT_TO=recipient@example.com
ALERT_FROM=your_email@gmail.com
    """)
    
    st.markdown("---")
    
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Base Directory:** {BASE_DIR}")
        st.write(f"**Output Directory:** {OUTPUT_DIR}")
        st.write(f"**Logs Directory:** {SNAP_DIR.parent}")
    with col2:
        st.write(f"**Max Image Size:** {MAX_IMAGE_SIZE / (1024*1024):.0f}MB")
        st.write(f"**Max Video Size:** {MAX_VIDEO_SIZE / (1024*1024):.0f}MB")
        st.write(f"**Database:** {BASE_DIR / 'logs' / 'violations.sqlite'}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B;'>SafetyEye AI • Production PPE Detection System</p>", unsafe_allow_html=True)
