"""
Dense Video Captioning - Streamlit Web Interface
=================================================
Professional web interface for the Dense Video Captioning system.
Supports video upload, demo videos, model selection, and result visualization.
"""

import os
import sys
import asyncio
import torch
torch.classes.__path__ = []

# Environment configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fix event loop issues on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import json
import time
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.config import Config
from src.scene_engine import SceneEngine
from src.model_factory import get_caption_engine
from src.merger import SceneMerger
from src.exporter import Exporter, ExportError

# ============================================
# CONFIGURATION
# ============================================
APP_TITLE = "Dense Video Captioning"
APP_DESCRIPTION = "AI-Powered Scene Detection & Caption Generation"
MAX_FILE_SIZE_MB = 500
SUPPORTED_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm']
DEMO_VIDEOS_PATH = Config.VIDEOS_DIR

# ============================================
# HELPER CLASSES
# ============================================

class DemoVideoFile:
    """Mock uploaded file object for demo videos."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self.size = os.path.getsize(file_path)
        self.type = f"video/{os.path.splitext(self.name)[1][1:]}"
    
    def getbuffer(self):
        with open(self.file_path, 'rb') as f:
            return f.read()

# ============================================
# SETUP FUNCTIONS
# ============================================

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Dense Video Captioning",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def add_custom_css():
    """Add professional custom CSS styling."""
    st.markdown("""
    <style>
    /* Professional sidebar styling */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 60%, rgba(102, 126, 234, 0.12) 100%) !important;
        min-height: 100vh;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Expander styling */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #667eea !important;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 1rem;
        background: rgba(102, 126, 234, 0.08);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.12);
        border-color: #7e3ff2;
        transform: translateY(-2px);
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        margin-bottom: 0.8rem;
        padding: 1rem;
    }

    /* Main header gradient */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Section headers with gradient underline */
    .section-header, .upload-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.3rem;
        text-align: left;
        position: relative;
    }
    
    .section-header::after, .upload-header::after {
        content: "";
        display: block;
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 3px;
        border-radius: 2px;
        background: linear-gradient(90deg, #7e3ff2 0%, #764ba2 60%, #667eea 100%);
    }

    /* Processing status badge */
    .processing-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Success message */
    .success-message {
        color: #28a745;
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Video navigation container */
    .navigation-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Video counter badge */
    .video-counter {
        font-size: 1rem;
        font-weight: 600;
        color: #7e3ff2;
        text-align: center;
        background: transparent;
        border: 2px solid #7e3ff2;
        border-radius: 8px;
        padding: 0.5rem 2.5rem;
        margin: 0 0.5rem;
    }
    
    /* Video preview container */
    .video-preview {
        max-width: 720px;
        margin: 0 auto;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    /* Statistics item */
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin: 0.3rem 0;
    }
    
    .stat-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        display: block;
        margin-bottom: 0.3rem;
    }
    
    .stat-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Scene items */
    .scene-item {
        padding: 0.5rem 0;
        margin: 0.3rem 0;
    }
    
    .scene-header {
        font-size: 1rem;
        font-weight: 600;
        color: #7e3ff2 !important;
        margin-bottom: 0.3rem;
    }
    
    .scene-caption {
        font-size: 1rem;
        line-height: 1.6;
        text-align: justify;
        font-style: italic;
        margin-left: 1rem;
    }
    
    /* Metric compact display */
    .metric-compact {
        text-align: center;
        padding: 0.3rem;
        margin: 0.1rem 0;
    }
    
    .metric-compact .metric-label {
        font-size: 0.8rem;
        color: #667eea !important;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }
    
    .metric-compact .metric-value {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Primary action buttons */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid transparent !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7e3ff2 0%, #a084e8 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(126, 63, 242, 0.6) !important;
    }
    
    /* Main container width */
    .main .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Sidebar section clean */
    .sidebar-section-clean {
        margin-bottom: 1rem;
        padding: 0;
        font-size: 0.95rem;
    }
    
    .sidebar-section-clean p {
        margin: 0 0 0.5rem 0;
        line-height: 1.4;
    }
    
    .sidebar-section-clean strong {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Reset button styling */
    .reset-btn-container {
        margin: 1rem 0;
        padding: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'uploaded_videos': [],
        'processed_videos': {},
        'current_video_index': 0,
        'processing_status': {},
        'demo_mode': False,
        'batch_processing': False,
        'selected_model': 'blip',
        'burn_subtitles': False,
        'caption_engine': None,
        'scene_merger': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================
# UTILITY FUNCTIONS
# ============================================

def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_demo_videos() -> List[str]:
    """Get list of demo videos from the videos directory."""
    demo_videos = []
    if os.path.exists(DEMO_VIDEOS_PATH):
        for f in os.listdir(DEMO_VIDEOS_PATH):
            if f.lower().endswith(tuple(SUPPORTED_FORMATS)):
                demo_videos.append(f)
    return sorted(demo_videos)

def load_all_demo_videos():
    """Load all demo videos into session state."""
    demo_videos = get_demo_videos()
    if demo_videos:
        demo_video_objects = []
        for video_name in demo_videos:
            video_path = os.path.join(DEMO_VIDEOS_PATH, video_name)
            demo_file = DemoVideoFile(video_path)
            demo_video_objects.append(demo_file)
        
        st.session_state.uploaded_videos = demo_video_objects
        st.session_state.demo_mode = True
        st.session_state.current_video_index = 0
        st.session_state.processed_videos = {}
        st.session_state.processing_status = {}

def reset_session():
    """Reset session state to allow reprocessing with different settings."""
    st.session_state.processed_videos = {}
    st.session_state.processing_status = {}
    st.session_state.current_video_index = 0
    # Force engine reinitialization on next run
    st.session_state.caption_engine = None
    st.session_state.scene_merger = None

def full_reset_session():
    """Full reset: clear everything and go back to initial state."""
    st.session_state.uploaded_videos = []
    st.session_state.processed_videos = {}
    st.session_state.current_video_index = 0
    st.session_state.processing_status = {}
    st.session_state.demo_mode = False
    st.session_state.batch_processing = False
    st.session_state.caption_engine = None
    st.session_state.scene_merger = None

def save_temp_file(uploaded_file, temp_dir: str = None) -> str:
    """Save uploaded file to temporary directory."""
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def get_or_create_engines(status_text=None, progress_bar=None):
    """Get or create caption engine and scene merger (cached)."""
    model = st.session_state.selected_model
    current_merge_threshold = Config.MERGE_THRESHOLD
    
    # Check if we need to reinitialize caption engine
    if (st.session_state.caption_engine is None or 
        getattr(st.session_state, '_current_model', None) != model):
        
        if status_text:
            status_text.text(f"🤖 Loading {model.upper()} model... (this may take a moment)")
        if progress_bar:
            progress_bar.progress(10)
        
        Config.CAPTION_MODEL_TYPE = model
        st.session_state.caption_engine = get_caption_engine(model)
        st.session_state._current_model = model
        
        if progress_bar:
            progress_bar.progress(12)
    
    # Check if we need to reinitialize scene merger (also when threshold changes)
    if (st.session_state.scene_merger is None or 
        getattr(st.session_state, '_current_merge_threshold', None) != current_merge_threshold):
        
        if status_text:
            status_text.text("🧠 Loading semantic model for scene merging...")
        if progress_bar:
            progress_bar.progress(14)
        
        st.session_state.scene_merger = SceneMerger(threshold=current_merge_threshold)
        st.session_state._current_merge_threshold = current_merge_threshold
    
    return st.session_state.caption_engine, st.session_state.scene_merger

# ============================================
# PROCESSING FUNCTIONS
# ============================================

def process_single_video(video_file, progress_container=None) -> bool:
    """Process a single video through the captioning pipeline."""
    try:
        model = st.session_state.selected_model
        burn_subs = st.session_state.burn_subtitles
        
        # Setup progress display
        if progress_container is None:
            progress_container = st.container()
        
        with progress_container:
            st.markdown(
                f'<div class="processing-status">🔄 Processing {video_file.name}...</div>',
                unsafe_allow_html=True
            )
            progress_bar = st.progress(0)
            status_text = st.empty()
        
            # Get or create engines (with loading indicator)
            status_text.text(f"🤖 Initializing {model.upper()} model...")
            progress_bar.progress(5)
            
            caption_engine, scene_merger = get_or_create_engines(status_text, progress_bar)
            
            # Get video path
            if hasattr(video_file, 'file_path'):  # Demo video
                video_path = video_file.file_path
            else:  # Uploaded video
                status_text.text("💾 Saving uploaded file...")
                progress_bar.progress(15)
                video_path = save_temp_file(video_file)
            
            video_name = video_file.name
            video_name_no_ext = os.path.splitext(video_name)[0]
            
            # Step 1: Scene Detection
            status_text.text("🔍 Detecting scenes...")
            progress_bar.progress(20)
            
            scene_engine = SceneEngine(video_path)
            scenes = scene_engine.detect_scenes()
            valid_scenes = [s for s in scenes if (s[1] - s[0]) >= 0.1]
            
            if not valid_scenes:
                st.error(f"❌ No valid scenes detected in {video_name}")
                return False
            
            # Step 2: Frame Extraction
            status_text.text(f"🎬 Extracting keyframes ({len(valid_scenes)} scenes)...")
            progress_bar.progress(35)
            
            visual_inputs = caption_engine.extract_frames_batch(video_path, valid_scenes)
            
            # Step 3: Caption Generation
            status_text.text("📝 Generating captions...")
            progress_bar.progress(50)
            
            raw_results = []
            for i, ((start, end), visual_input) in enumerate(zip(valid_scenes, visual_inputs)):
                if visual_input is None:
                    continue
                
                description = caption_engine.generate_caption(visual_input)
                if description:
                    raw_results.append({
                        "Scene ID": i + 1,
                        "Start": start,
                        "End": end,
                        "Duration": round(end - start, 2),
                        "Description": description
                    })
                
                # Update progress
                pct = 50 + int(((i + 1) / len(valid_scenes)) * 25)
                progress_bar.progress(min(pct, 75))
                status_text.text(f"📝 Generating captions... ({i+1}/{len(valid_scenes)})")
            
            if not raw_results:
                st.error(f"❌ No captions generated for {video_name}")
                return False
            
            # Step 4: Scene Merging
            status_text.text("🔀 Merging similar scenes...")
            progress_bar.progress(80)
            
            if Config.ENABLE_SCENE_MERGING:
                final_results = scene_merger.merge_scenes(raw_results)
            else:
                final_results = raw_results
            
            # Step 5: Export Results
            status_text.text("💾 Saving results...")
            progress_bar.progress(90)
            
            try:
                created_files = Exporter.save_results(
                    final_results, video_name, video_path, model, burn_subs
                )
            except ExportError as e:
                st.error(f"❌ Export failed: {e}")
                return False
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Calculate video duration
            video_duration = max([s['End'] for s in final_results]) if final_results else 0
            
            # Store results in session state
            st.session_state.processed_videos[video_file.name] = {
                'name': video_name,
                'results': final_results,
                'raw_results': raw_results,
                'video_path': video_path,
                'created_files': created_files,
                'model': model,
                'video_duration': video_duration,
                'has_subtitles': burn_subs and 'video' in created_files
            }
            
            st.session_state.processing_status[video_file.name] = 'completed'
            
            time.sleep(1)
        
        # Clear progress container
        progress_container.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing {video_file.name}: {str(e)}")
        st.session_state.processing_status[video_file.name] = 'error'
        return False

def process_all_videos():
    """Process all uploaded/demo videos sequentially."""
    st.session_state.batch_processing = True
    total_videos = len(st.session_state.uploaded_videos)
    
    main_progress = st.container()
    
    with main_progress:
        st.markdown(
            '<div class="processing-status">🚀 Batch Processing... Please wait.</div>',
            unsafe_allow_html=True
        )
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        for idx, video_file in enumerate(st.session_state.uploaded_videos):
            if st.session_state.processing_status.get(video_file.name) == 'completed':
                overall_progress.progress((idx + 1) / total_videos)
                overall_status.text(f"✅ {idx + 1}/{total_videos}: {video_file.name} (Already processed)")
                continue
            
            overall_status.text(f"🔄 Processing {idx + 1}/{total_videos}: {video_file.name}")
            st.session_state.processing_status[video_file.name] = 'processing'
            
            video_progress = st.container()
            success = process_single_video(video_file, video_progress)
            
            if success:
                st.session_state.processing_status[video_file.name] = 'completed'
            else:
                st.session_state.processing_status[video_file.name] = 'error'
            
            overall_progress.progress((idx + 1) / total_videos)
        
        overall_status.text(f"✅ Batch processing complete! ({total_videos} videos)")
        time.sleep(2)
        main_progress.empty()
    
    st.session_state.batch_processing = False
    st.session_state.current_video_index = 0

# ============================================
# SIDEBAR
# ============================================

def setup_sidebar():
    """Setup the sidebar with settings and information."""
    
    # ============================================
    # QUICK START (FIRST)
    # ============================================
    with st.sidebar.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p><strong>1.</strong> Select a caption model</p>
        <p><strong>2.</strong> Experiment with configuration settings</p>
        <p><strong>3.</strong> Upload videos or use demos</p>
        <p><strong>4.</strong> Click process to generate captions</p>
        <p><strong>5.</strong> View results and download outputs</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # ============================================
    # SETTINGS SECTION
    # ============================================
    st.sidebar.markdown("### ⚙️ Settings")
    
    # MODEL SELECTION
    model_options = {
        'blip': 'BLIP (Fast, Frame-based)',
        'git': 'GIT (Video, Multi-frame)',
        'qwen': 'Qwen2-VL (VLM, Best Quality)'
    }
    
    selected = st.sidebar.selectbox(
        "🤖 Caption Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.selected_model),
        help="Select the AI model for caption generation"
    )
    st.session_state.selected_model = selected
    
    # Subtitle burning option
    st.session_state.burn_subtitles = st.sidebar.checkbox(
        "🔥 Burn Subtitles into Video",
        value=st.session_state.burn_subtitles,
        help="Create a copy of the video with embedded subtitles (requires FFmpeg)"
    )
    
    st.sidebar.markdown("---")
    
    # ============================================
    # RESET / REPROCESS CONTROLS
    # ============================================
    has_processed = len(st.session_state.processed_videos) > 0
    has_videos = len(st.session_state.uploaded_videos) > 0
    
    if has_processed or has_videos:
        st.sidebar.markdown("### 🔄 Session Controls")
        
        if has_processed:
            if st.sidebar.button("🔄 Clear Results & Reprocess", use_container_width=True,
                                help="Keep current videos but clear all results. Useful for trying a different model or settings."):
                reset_session()
                st.rerun()
        
        if has_videos:
            if st.sidebar.button("↩️ Back to Start", use_container_width=True,
                                help="Clear everything and go back to the upload screen."):
                full_reset_session()
                st.rerun()
        
        st.sidebar.markdown("---")
    
    # ============================================
    # CONFIGURATION SECTION
    # ============================================
    st.sidebar.markdown("### 🛠️ Configuration")
    
    # SCENE DETECTION SETTINGS
    with st.sidebar.expander("🎬 Scene Detection", expanded=False):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p>Configure how scenes are detected in videos using content-aware analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        content_threshold = st.slider(
            "Content Threshold",
            min_value=10.0,
            max_value=50.0,
            value=float(Config.CONTENT_THRESHOLD),
            step=1.0,
            help="Higher values = fewer scene cuts detected. Lower values = more sensitive detection."
        )
        Config.CONTENT_THRESHOLD = content_threshold
        
        min_scene_len = st.slider(
            "Min Scene Length (frames)",
            min_value=5,
            max_value=60,
            value=int(Config.MIN_SCENE_LEN),
            step=1,
            help="Minimum number of frames required for a valid scene."
        )
        Config.MIN_SCENE_LEN = min_scene_len
    
    # SEMANTIC MERGING SETTINGS
    with st.sidebar.expander("🔀 Semantic Merging", expanded=False):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p>Configure how similar consecutive scenes are merged based on caption similarity.</p>
        </div>
        """, unsafe_allow_html=True)
        
        merge_threshold = st.slider(
            "Merge Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(Config.MERGE_THRESHOLD),
            step=0.05,
            help="Cosine similarity threshold for merging scenes. Higher = stricter merging (fewer merges)."
        )
        Config.MERGE_THRESHOLD = merge_threshold
    
    # KEYFRAME EXTRACTION (BLIP ONLY)
    with st.sidebar.expander("🖼️ Keyframe Extraction (BLIP)", expanded=False):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p>⚠️ <strong>BLIP model only.</strong> Configure how the best frame is selected from each scene.</p>
        </div>
        """, unsafe_allow_html=True)
        
        weight_sharp = st.slider(
            "Sharpness Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(Config.WEIGHT_SHARP),
            step=0.05,
            help="Weight for sharpness metric in keyframe selection."
        )
        
        # Auto-calculate entropy weight
        weight_entropy = 1.0 - weight_sharp
        
        st.markdown(f"""
        <div class="sidebar-section-clean">
        <p><strong>Entropy Weight:</strong> {weight_entropy:.2f} (auto-calculated)</p>
        </div>
        """, unsafe_allow_html=True)
        
        Config.WEIGHT_SHARP = weight_sharp
        Config.WEIGHT_ENTROPY = weight_entropy
    
    # ABLATION STUDY FLAGS
    with st.sidebar.expander("🔬 Ablation Study", expanded=False):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p>Toggle system components on/off for ablation experiments.</p>
        </div>
        """, unsafe_allow_html=True)
        
        enable_merging = st.checkbox(
            "Enable Scene Merging",
            value=Config.ENABLE_SCENE_MERGING,
            help="When disabled, scenes are not merged based on semantic similarity."
        )
        Config.ENABLE_SCENE_MERGING = enable_merging
        
        enable_keyframe = st.checkbox(
            "Enable Smart Keyframe Selection",
            value=Config.ENABLE_KEYFRAME_SELECTION,
            help="When disabled, uses middle frame instead of sharpness/entropy-based selection. (BLIP only)"
        )
        Config.ENABLE_KEYFRAME_SELECTION = enable_keyframe
        
        if not enable_keyframe:
            st.info("📍 Using mid-frame selection (baseline mode)")
        
        if not enable_merging:
            st.info("🔓 Scene merging disabled (raw output)")
    
    st.sidebar.markdown("---")
    
    # ============================================
    # INFORMATION SECTION
    # ============================================
    st.sidebar.markdown("### 📚 Information")
    
    with st.sidebar.expander("🤖 AI Models", expanded=False):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p><strong>BLIP (Salesforce)</strong><br>
        Fast frame-based captioning with beam search decoding. Uses smart keyframe selection.</p>
        <p><strong>GIT(Microsoft)</strong><br>
        Video-aware model processing 6 frames per scene for temporal understanding.</p>
        <p><strong>Qwen2-VL (Alibaba)</strong><br>
        State-of-the-art Vision-Language Model with 4-bit quantization. Best quality but slower.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar.expander("🔄 Processing Pipeline", expanded=False):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p><strong>Scene Detection</strong><br>
        PySceneDetect with adaptive thresholding and fallback mode</p>
        <p><strong>Keyframe Extraction</strong><br>
        Smart selection using sharpness & entropy metrics (BLIP) or multi-frame sampling (GIT/Qwen)</p>
        <p><strong>Caption Generation</strong><br>
        Deep learning model inference on GPU/CPU</p>
        <p><strong>Semantic Merging</strong><br>
        SentenceTransformer-based scene consolidation using cosine similarity</p>
        <p><strong>Export</strong><br>
        JSON, CSV, SRT, and optional subtitled video</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar.expander("📋 Current Config", expanded=False):
        st.markdown(f"""
        <div class="sidebar-section-clean">
        <p><strong>Device</strong>: {Config.DEVICE.upper()}</p>
        <p><strong>Scene Threshold</strong>: {Config.CONTENT_THRESHOLD}</p>
        <p><strong>Min Scene Frames</strong>: {Config.MIN_SCENE_LEN}</p>
        <p><strong>Merge Threshold</strong>: {Config.MERGE_THRESHOLD}</p>
        <p><strong>Keyframe Weights</strong>: Sharp={Config.WEIGHT_SHARP:.2f}, Entropy={Config.WEIGHT_ENTROPY:.2f}</p>
        <p><strong>Scene Merging</strong>: {'✅ Enabled' if Config.ENABLE_SCENE_MERGING else '❌ Disabled'}</p>
        <p><strong>Smart Keyframes</strong>: {'✅ Enabled' if Config.ENABLE_KEYFRAME_SELECTION else '❌ Disabled'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar.expander("👨‍💻 About", expanded=False):
        st.markdown("""
        <div class="sidebar-section-clean">
        <p><strong>Project</strong><br>
        Dense Video Captioning System</p>
        <p><strong>Purpose</strong><br>
        Academic Research & Thesis Implementation</p>
        <p><strong>Technologies</strong><br>
        PyTorch, Transformers, Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# DISPLAY FUNCTIONS
# ============================================

def display_demo_videos():
    """Display demo videos selection button."""
    demo_videos = get_demo_videos()
    
    if not demo_videos:
        st.info(f"📁 No demo videos found in {DEMO_VIDEOS_PATH}")
        st.markdown("💡 Add video files to the videos folder to see them here!")
        return
    
    st.markdown('<div class="section-header">🎬 Demo Videos</div>', unsafe_allow_html=True)
    
    if st.button(
        f"🎥 Load Demo Videos ({len(demo_videos)} available)",
        key="load_demos",
        type="primary",
        use_container_width=True
    ):
        load_all_demo_videos()
        st.rerun()

def display_video_navigation():
    """Display video navigation controls."""
    if not st.session_state.uploaded_videos:
        return
    
    total_videos = len(st.session_state.uploaded_videos)
    current_idx = st.session_state.current_video_index
    
    st.markdown('<div class="navigation-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", disabled=(current_idx == 0), key="first_btn", use_container_width=True):
            st.session_state.current_video_index = 0
            st.rerun()
    
    with col2:
        if st.button("⬅️ Prev", disabled=(current_idx == 0), key="prev_btn", use_container_width=True):
            st.session_state.current_video_index = max(0, current_idx - 1)
            st.rerun()
    
    with col3:
        demo_tag = " (Demo)" if st.session_state.demo_mode else ""
        st.markdown(f'''
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <div class="video-counter">Video {current_idx + 1} of {total_videos}{demo_tag}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        if st.button("Next ➡️", disabled=(current_idx >= total_videos - 1), key="next_btn", use_container_width=True):
            st.session_state.current_video_index = min(total_videos - 1, current_idx + 1)
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", disabled=(current_idx >= total_videos - 1), key="last_btn", use_container_width=True):
            st.session_state.current_video_index = total_videos - 1
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_current_video():
    """Display the current video and its processing controls/results."""
    if not st.session_state.uploaded_videos:
        return
    
    current_idx = st.session_state.current_video_index
    current_video = st.session_state.uploaded_videos[current_idx]
    file_size_mb = current_video.size / (1024 * 1024)
    
    # Video metadata
    _, col2, col3, col4, _ = st.columns([2, 1, 1, 1, 2])
    
    with col2:
        st.markdown(f'''
        <div class="metric-compact">
            <div class="metric-label">📁 File Name</div>
            <div class="metric-value">{current_video.name}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-compact">
            <div class="metric-label">📏 File Size</div>
            <div class="metric-value">{file_size_mb:.2f} MB</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-compact">
            <div class="metric-label">🤖 Model</div>
            <div class="metric-value">{st.session_state.selected_model.upper()}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Check if video is processed
    if current_video.name in st.session_state.processed_videos:
        display_processed_video(st.session_state.processed_videos[current_video.name])
    else:
        display_unprocessed_video(current_video)

def display_unprocessed_video(video_file):
    """Display unprocessed video with processing controls."""
    
    # Video preview (compact)
    _, video_col, _ = st.columns([1, 3, 1])
    with video_col:
        st.markdown('<div style="text-align: center;"><h3>📺 Video Preview</h3></div>', unsafe_allow_html=True)
        st.markdown('<div class="video-preview">', unsafe_allow_html=True)
        if hasattr(video_file, 'file_path'):
            st.video(video_file.file_path)
        else:
            st.video(video_file)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing status check
    status = st.session_state.processing_status.get(video_file.name)
    if status == 'processing':
        st.warning("🔄 This video is currently being processed...")
        return
    elif status == 'error':
        st.error("❌ Processing failed for this video. Try again.")
    
    # Processing buttons
    st.markdown('<div style="margin-top: 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    total_videos = len(st.session_state.uploaded_videos)
    unprocessed = sum(1 for v in st.session_state.uploaded_videos 
                      if st.session_state.processing_status.get(v.name) != 'completed')
    
    if total_videos > 1 and unprocessed > 1:
        btn_col1, _, btn_col2 = st.columns([2, 0.3, 2])
        
        with btn_col1:
            if st.button("🚀 Process Current Video", key=f"process_{st.session_state.current_video_index}", 
                        type="primary", use_container_width=True):
                st.session_state.processing_status[video_file.name] = 'processing'
                progress_container = st.container()
                if process_single_video(video_file, progress_container):
                    st.rerun()
        
        with btn_col2:
            if st.button(f"🎬 Process All Videos ({unprocessed} remaining)", 
                        key="process_all", type="primary", use_container_width=True):
                process_all_videos()
                st.rerun()
    else:
        _, btn_col, _ = st.columns([1.5, 3, 1.5])
        with btn_col:
            if st.button("🚀 Process Video", key=f"process_{st.session_state.current_video_index}", 
                        type="primary", use_container_width=True):
                st.session_state.processing_status[video_file.name] = 'processing'
                progress_container = st.container()
                if process_single_video(video_file, progress_container):
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_processed_video(processed_data: Dict[str, Any]):
    """Display processed video with results."""
    st.markdown('<div class="success-message">🎉 Video captioning completed successfully!</div>', 
                unsafe_allow_html=True)
    
    results = processed_data['results']
    video_path = processed_data['video_path']
    created_files = processed_data.get('created_files', {})
    
    # Video display (compact)
    _, video_col, _ = st.columns([1, 3, 1])
    with video_col:
        subtitle_tag = "📺 Subtitled Video" if processed_data.get('has_subtitles') else "📺 Original Video"
        st.markdown(f'<div style="text-align: center;"><h3>{subtitle_tag}</h3></div>', unsafe_allow_html=True)
        
        display_video = created_files.get('video', video_path)
        if display_video and os.path.exists(display_video):
            st.markdown('<div class="video-preview">', unsafe_allow_html=True)
            st.video(display_video)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Video Statistics
    st.markdown('<div class="section-header">📊 Video Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_duration = processed_data.get('video_duration', 0)
    avg_scene_len = sum([s.get('Duration', s['End'] - s['Start']) for s in results]) / len(results) if results else 0
    total_words = sum([len(s.get('Description', '').split()) for s in results])
    
    with col1:
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{len(results)}</span>
            <div class="stat-label">🎬 Total Scenes</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{total_duration:.1f}s</span>
            <div class="stat-label">⏱️ Duration</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{avg_scene_len:.1f}s</span>
            <div class="stat-label">📊 Avg Scene</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{total_words}</span>
            <div class="stat-label">📝 Total Words</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Scene Descriptions
    st.markdown('<div class="section-header">🎬 Scene Descriptions</div>', unsafe_allow_html=True)
    
    for i, scene in enumerate(results):
        start = scene.get('Start', 0)
        end = scene.get('End', 0)
        duration = scene.get('Duration', end - start)
        caption = scene.get('Description', '')
        
        st.markdown(f'''
        <div class="scene-item">
            <div class="scene-header">
                🎬 Scene {i+1}: {format_timestamp(start)} - {format_timestamp(end)} ({duration:.1f}s)
            </div>
            <div class="scene-caption">{caption}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Download Section
    st.markdown('<div class="section-header">📥 Download Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    video_name_no_ext = os.path.splitext(processed_data['name'])[0]
    
    with col1:
        # JSON download
        json_data = {
            "video": processed_data['name'],
            "model": processed_data.get('model', 'unknown'),
            "total_scenes": len(results),
            "duration": total_duration,
            "scenes": results
        }
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 JSON Data",
            data=json_str,
            file_name=f"{video_name_no_ext}_captions.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Subtitled video download
        subtitled_path = created_files.get('video')
        if subtitled_path and os.path.exists(subtitled_path):
            with open(subtitled_path, "rb") as f:
                st.download_button(
                    label="🎥 Subtitled Video",
                    data=f.read(),
                    file_name=f"{video_name_no_ext}_subtitled.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        else:
            st.button(
                label="🎥 No Subtitled Video",
                disabled=True,
                use_container_width=True
            )
    
    with col3:
        # SRT download
        srt_content = ""
        for i, scene in enumerate(results):
            start = scene.get('Start', 0)
            end = scene.get('End', 0)
            caption = scene.get('Description', '')
            
            start_ts = Exporter.format_timestamp(start)
            end_ts = Exporter.format_timestamp(end)
            
            srt_content += f"{i+1}\n"
            srt_content += f"{start_ts} --> {end_ts}\n"
            srt_content += f"{caption}\n\n"
        
        st.download_button(
            label="🎬 SRT Subtitles",
            data=srt_content,
            file_name=f"{video_name_no_ext}_subtitles.srt",
            mime="text/plain",
            use_container_width=True
        )

def display_features():
    """Display feature highlights when no videos are loaded."""
    st.markdown("### 🌟 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **AI-Powered Analysis:**
        - 🎯 Automatic scene detection
        - 🖼️ Smart keyframe extraction
        - 🧠 Deep learning captioning
        - 🔀 Semantic scene merging
        """)
    
    with col2:
        st.markdown("""
        **Output Formats:**
        - 📄 JSON with full metadata
        - 📊 CSV analysis files
        - 🎬 SRT subtitle files
        - 🎥 Subtitled video export
        """)

# ============================================
# MAIN CONTENT
# ============================================

def main_content_area():
    """Main content area with upload and video display."""
    
    is_processing = any(s == 'processing' for s in st.session_state.processing_status.values())
    
    if not is_processing and not st.session_state.batch_processing:
        setup_sidebar()
    
    # Upload section
    st.markdown('<div class="upload-header">📁 Upload Videos</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose one or more video files",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS).upper()}. Max size: {MAX_FILE_SIZE_MB}MB",
        key="video_uploader"
    )
    
    # Handle new uploads
    new_file_names = {f.name for f in uploaded_files}
    old_file_names = {f.name for f in st.session_state.get('uploaded_videos', []) 
                      if not st.session_state.get('demo_mode', False)}
    
    if new_file_names != old_file_names and uploaded_files:
        # Filter by file size
        valid_files = [f for f in uploaded_files if f.size <= MAX_FILE_SIZE_MB * 1024 * 1024]
        if len(valid_files) != len(uploaded_files):
            st.error(f"❌ Some files exceed {MAX_FILE_SIZE_MB}MB limit and were excluded.")
        
        # Update session state
        valid_names = {f.name for f in valid_files}
        
        st.session_state.uploaded_videos = valid_files
        st.session_state.demo_mode = False
        st.session_state.processed_videos = {
            k: v for k, v in st.session_state.processed_videos.items() if k in valid_names
        }
        st.session_state.processing_status = {
            k: v for k, v in st.session_state.processing_status.items() if k in valid_names
        }
        st.session_state.current_video_index = 0
        st.rerun()
    
    # Display content based on state
    if st.session_state.get('uploaded_videos'):
        display_video_navigation()
        display_current_video()
    else:
        display_demo_videos()
        st.info("👆 Upload your own videos or try the demo videos to begin processing")
        display_features()

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point."""
    setup_page_config()
    add_custom_css()
    initialize_session_state()
    
    # Initialize directories
    try:
        Config.initialize_directories()
    except Exception as e:
        st.error(f"❌ Failed to initialize directories: {e}")
        return
    
    # Header
    st.markdown(f'<h1 class="main-header">🎬 {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align: center; font-size: 1.1rem; color: #666;'>{APP_DESCRIPTION}</p>",
        unsafe_allow_html=True
    )
    
    # Main content
    main_content_area()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, PyTorch, and Transformers</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()