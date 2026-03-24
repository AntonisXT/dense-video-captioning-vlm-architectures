import os
import torch
from typing import List, Dict, Any

class Config:
    """
    Centralized configuration for Dense Video Captioning system.
    
    This class manages:
    - Directory paths (videos, results, logs, evaluation)
    - Model selection and parameters
    - Scene detection settings
    - Semantic merging thresholds
    - Evaluation parameters
    - Ablation study flags
    
    Usage:
        # At program startup
        Config.initialize_directories()
        Config.validate()
        
        # Access settings
        model = Config.CAPTION_MODEL_TYPE
        device = Config.DEVICE
    """
    
    # ========================================
    # DIRECTORY PATHS
    # ========================================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    VIDEOS_DIR = os.path.join(DATA_DIR, 'videos')
    RESULTS_DIR = os.path.join(DATA_DIR, 'results')
    LOGS_DIR = os.path.join(DATA_DIR, 'logs')
    
    # Evaluation folders
    EVAL_DIR = os.path.join(BASE_DIR, 'evaluation')
    EVAL_REPORTS_DIR = os.path.join(EVAL_DIR, 'reports')
    EVAL_E2E_DIR = os.path.join(EVAL_REPORTS_DIR, 'e2e')
    EVAL_ORACLE_DIR = os.path.join(EVAL_REPORTS_DIR, 'oracle')
    EVAL_OFFLINE_DIR = os.path.join(EVAL_REPORTS_DIR, 'offline')
    
    # Visualization output
    EVAL_PLOTS_DIR = os.path.join(EVAL_REPORTS_DIR, 'plots')
    EVAL_COMPARISONS_DIR = os.path.join(EVAL_PLOTS_DIR, 'comparisons')

    # ========================================
    # MODEL CONFIGURATION
    # ========================================
    # Options: "blip" (Frame-based), "git" (Video), "qwen" (VLM)
    CAPTION_MODEL_TYPE = "blip"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model names
    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    GIT_MODEL_NAME = "microsoft/git-large-vatex"
    QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

    # ========================================
    # GENERATION SETTINGS
    # ========================================
    # BLIP
    BLIP_NUM_BEAMS = 5
    BLIP_MIN_LENGTH = 10
    BLIP_MAX_LENGTH = 60

    # GIT
    GIT_MIN_LENGTH = 10
    GIT_MAX_LENGTH = 60

    # Qwen
    QWEN_MAX_LENGTH = 200
    QWEN_DO_SAMPLE = True
    QWEN_TEMPERATURE = 0.2
    QWEN_TOP_P = 0.9

    # ========================================
    # SEMANTIC MERGING SETTINGS
    # ========================================
    SIMILARITY_MODEL = "all-MiniLM-L6-v2"
    MERGE_THRESHOLD = 0.75          # Cosine similarity threshold for merging scenes
    
    # ========================================
    # SCENE DETECTION SETTINGS
    # ========================================
    CONTENT_THRESHOLD = 27.0        # Standard sensitivity
    MIN_SCENE_LEN = 25              # Minimum scene length in frames
    
    # Fallback mode (higher sensitivity)
    CONTENT_THRESHOLD_FALLBACK = 17.0
    MIN_SCENE_LEN_FALLBACK = 8

    # ========================================
    # KEYFRAME EXTRACTION SETTINGS
    # ========================================
    NUM_SAMPLES = 5                 # Number of candidate frames to sample
    WEIGHT_SHARP = 0.70             # Weight for sharpness metric
    WEIGHT_ENTROPY = 0.30           # Weight for entropy metric

    # ========================================
    # EVALUATION SETTINGS
    # ========================================
    EVAL_IOU_THRESHOLD = 0.3        # IoU threshold for matching predictions to GT

    # ========================================
    # ABLATION STUDY FLAGS
    # ========================================
    # Enable/disable specific components for ablation studies
    ENABLE_SCENE_MERGING = True             # Merge semantically similar scenes 
    ENABLE_KEYFRAME_SELECTION = True        # Use smart keyframe extraction vs random
    
    # Ablation experiment name (for tracking results)
    ABLATION_EXPERIMENT = "full_system"      # Options: "full_system", "no_merging", etc.

    # ========================================
    # LOGGING & DEBUGGING
    # ========================================
    LOG_LEVEL = "INFO"              # Options: DEBUG, INFO, WARNING, ERROR
    ENABLE_PROGRESS_BARS = True     # Show tqdm progress bars
    SAVE_INTERMEDIATE_FRAMES = False # Save extracted keyframes (for debugging)

    # ========================================
    # INITIALIZATION & VALIDATION
    # ========================================
    
    @classmethod
    def initialize_directories(cls) -> None:
        """
        Create all required directories if they don't exist.
        
        Call this once at program startup (e.g., in main.py or benchmark.py).
        
        Example:
            >>> Config.initialize_directories()
            >>> # Now all paths are ready
        """
        directories = [
            cls.VIDEOS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR,
            cls.EVAL_REPORTS_DIR,
            cls.EVAL_E2E_DIR,
            cls.EVAL_ORACLE_DIR,
            cls.EVAL_OFFLINE_DIR,
            cls.EVAL_PLOTS_DIR,
            cls.EVAL_COMPARISONS_DIR,
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create directory {directory}: {e}")
        
        print(f"✅ Initialized {len(directories)} directories")

    @classmethod
    def validate(cls) -> None:
        """
        Validate all configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
            
        Example:
            >>> Config.validate()  # Raises if config is invalid
        """
        errors: List[str] = []
        
        # --- Model Selection ---
        valid_models = ["blip", "git", "qwen"]
        if cls.CAPTION_MODEL_TYPE.lower() not in valid_models:
            errors.append(
                f"CAPTION_MODEL_TYPE must be one of {valid_models}, "
                f"got '{cls.CAPTION_MODEL_TYPE}'"
            )
        
        # --- Merge Thresholds ---
        if not (0.0 <= cls.MERGE_THRESHOLD <= 1.0):
            errors.append(
                f"MERGE_THRESHOLD must be in [0, 1], got {cls.MERGE_THRESHOLD}"
            )
        
        # --- Scene Detection ---
        if cls.CONTENT_THRESHOLD <= 0:
            errors.append(
                f"CONTENT_THRESHOLD must be > 0, got {cls.CONTENT_THRESHOLD}"
            )
        
        if cls.MIN_SCENE_LEN < 1:
            errors.append(
                f"MIN_SCENE_LEN must be >= 1, got {cls.MIN_SCENE_LEN}"
            )
        
        # --- Keyframe Extraction ---
        if cls.NUM_SAMPLES < 1:
            errors.append(
                f"NUM_SAMPLES must be >= 1, got {cls.NUM_SAMPLES}"
            )
        
        weight_sum = cls.WEIGHT_SHARP + cls.WEIGHT_ENTROPY
        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
            errors.append(
                f"WEIGHT_SHARP + WEIGHT_ENTROPY must equal 1.0, "
                f"got {weight_sum:.4f}"
            )
        
        if not (0.0 <= cls.WEIGHT_SHARP <= 1.0):
            errors.append(f"WEIGHT_SHARP must be in [0, 1], got {cls.WEIGHT_SHARP}")
        
        if not (0.0 <= cls.WEIGHT_ENTROPY <= 1.0):
            errors.append(f"WEIGHT_ENTROPY must be in [0, 1], got {cls.WEIGHT_ENTROPY}")
        
        # --- Evaluation ---
        if not (0.0 <= cls.EVAL_IOU_THRESHOLD <= 1.0):
            errors.append(
                f"EVAL_IOU_THRESHOLD must be in [0, 1], "
                f"got {cls.EVAL_IOU_THRESHOLD}"
            )
        
        # --- Generation Settings ---
        if cls.BLIP_MIN_LENGTH > cls.BLIP_MAX_LENGTH:
            errors.append(
                f"BLIP_MIN_LENGTH ({cls.BLIP_MIN_LENGTH}) must be <= "
                f"BLIP_MAX_LENGTH ({cls.BLIP_MAX_LENGTH})"
            )
        
        if cls.GIT_MIN_LENGTH > cls.GIT_MAX_LENGTH:
            errors.append(
                f"GIT_MIN_LENGTH ({cls.GIT_MIN_LENGTH}) must be <= "
                f"GIT_MAX_LENGTH ({cls.GIT_MAX_LENGTH})"
            )
        
        # --- Device Check ---
        if cls.DEVICE == "cuda" and not torch.cuda.is_available():
            errors.append("DEVICE set to 'cuda' but CUDA is not available")
        
        # --- Raise all errors at once ---
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  ❌ {err}" for err in errors
            )
            raise ValueError(error_msg)
        
        print("✅ Configuration validated successfully")

    @classmethod
    def get_ablation_suffix(cls) -> str:
        """
        Generate a suffix for output files based on ablation settings.
        
        Returns:
            str: Suffix like "_no_merging" or "_no_cleaning"
            
        Example:
            >>> Config.ENABLE_SCENE_MERGING = False
            >>> Config.get_ablation_suffix()
            '_no_merging'
        """
        if cls.ABLATION_EXPERIMENT != "full_system":
            return f"_{cls.ABLATION_EXPERIMENT}"
        
        disabled_features = []
        
        if not cls.ENABLE_SCENE_MERGING:
            disabled_features.append("no_merging")        
        
        if not cls.ENABLE_KEYFRAME_SELECTION:
            disabled_features.append("random_frames")
        
        if disabled_features:
            return "_" + "_".join(disabled_features)
        
        return ""

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Export configuration as dictionary (safe for JSON serialization).

        Keeps only UPPERCASE configuration constants and converts any
        non-JSON-serializable values to strings.
        """
        import json

        def is_jsonable(x: Any) -> bool:
            try:
                json.dumps(x)
                return True
            except Exception:
                return False

        def safe_value(x: Any) -> Any:
            # Exclude methods / classmethods / staticmethods and callables
            if isinstance(x, (classmethod, staticmethod)):
                return None
            if callable(x):
                return None

            # Keep JSON-serializable primitives/containers as-is
            if is_jsonable(x):
                return x

            # Fallback: stringify anything else (Path, devices, objects, etc.)
            try:
                return str(x)
            except Exception:
                return None

        cfg: Dict[str, Any] = {}

        for key, value in vars(cls).items():
            # keep only constants
            if not isinstance(key, str) or not key.isupper():
                continue

            v = safe_value(value)
            if v is None:
                continue

            cfg[key] = v

        return cfg


    @classmethod
    def print_summary(cls) -> None:
        """
        Print a formatted summary of current configuration.
        
        Useful for logging at program startup.
        """
        print("\n" + "="*60)
        print("⚙️  CONFIGURATION SUMMARY")
        print("="*60)
        print(f"🎯 Model:              {cls.CAPTION_MODEL_TYPE.upper()}")
        print(f"💻 Device:             {cls.DEVICE}")
        print(f"🔀 Scene Merging:      {'Enabled' if cls.ENABLE_SCENE_MERGING else 'Disabled'}")        
        print(f"🎯 Merge Threshold:    {cls.MERGE_THRESHOLD}")
        print(f"📏 IoU Threshold:      {cls.EVAL_IOU_THRESHOLD}")
        print(f"🎬 Scene Detection:    {cls.CONTENT_THRESHOLD}")
        
        if cls.ABLATION_EXPERIMENT != "full_system":
            print(f"🔬 Ablation Mode:      {cls.ABLATION_EXPERIMENT}")
        
        print("="*60 + "\n")


# Example usage in main.py or benchmark.py:
if __name__ == "__main__":
    # Initialize and validate at startup
    try:
        Config.initialize_directories()
        Config.validate()
        Config.print_summary()
        
        print("✅ Config ready for use!")
        
    except (ValueError, RuntimeError) as e:
        print(f"❌ Configuration Error:\n{e}")
        exit(1)