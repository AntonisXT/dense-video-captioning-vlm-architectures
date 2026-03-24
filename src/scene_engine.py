from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
from .config import Config
from .logger import log

class SceneEngine:
    """
    Handles video scene detection using adaptive thresholding logic.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path

    def _run_detection(self, threshold: float, min_len: float) -> list:
        """Executes content-aware scene detection with specified parameters."""
        video = open_video(self.video_path)
        scene_manager = SceneManager()
        
        detector = ContentDetector(threshold=threshold, min_scene_len=min_len)
        scene_manager.add_detector(detector)
        
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        
        if scene_list:
            return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
        return []

    def detect_scenes(self) -> list:
        """
        Main detection pipeline with automatic fallback logic.
        
        Process:
            1. Attempt standard detection using default configuration.
            2. If it fails (no scenes or single scene), retry with higher sensitivity (fallback).
            3. If both fail, treat the entire video as a single scene.
        """
        # --- ATTEMPT 1: STANDARD MODE ---
        log.info(f"   🎥 Detecting scenes (Standard Mode: {Config.CONTENT_THRESHOLD})...")
        scenes = self._run_detection(
            threshold=Config.CONTENT_THRESHOLD, 
            min_len=Config.MIN_SCENE_LEN
        )

        # Determine if retry is needed (0 cuts found means usually failure for long videos)
        needs_retry = not scenes or len(scenes) == 1

        # --- ATTEMPT 2: FALLBACK / SENSITIVE MODE ---
        if needs_retry:
            log.warning(f"   ⚠️  Insufficient scenes found. Retrying with high sensitivity (Threshold: {Config.CONTENT_THRESHOLD_FALLBACK})...")
            
            sensitive_scenes = self._run_detection(
                threshold=Config.CONTENT_THRESHOLD_FALLBACK,
                min_len=Config.MIN_SCENE_LEN_FALLBACK
            )
            
            if sensitive_scenes:
                log.info(f"   ✅ Sensitive scan found {len(sensitive_scenes)} scenes.")
                scenes = sensitive_scenes
            else:
                log.info("   ℹ️  Video appears to be a single continuous shot.")

        # Post-Processing: Default to full video if detection fails completely
        if not scenes:
            scenes = [(0.0, self._get_video_duration())]
            
        return scenes

    def _get_video_duration(self) -> float:
        """Helper to retrieve total video duration via OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        return frames / fps if fps > 0 else 0.0