from colorama import Fore
from src.config import Config

def get_caption_engine(model_type: str = None):
    """
    Factory method to instantiate the appropriate captioning engine.
    
    Args:
        model_type (str, optional): The model identifier (blip, git, qwen). 
                                    Defaults to Config.CAPTION_MODEL_TYPE.
    """
    target_model = model_type if model_type else Config.CAPTION_MODEL_TYPE
    target_model = target_model.lower().strip()

    print(Fore.BLUE + f"🏗️  Initializing Caption Engine: {target_model.upper()}...")

    if target_model == "blip":
        from src.engines.blip import CaptionEngine
        return CaptionEngine()
    
    elif target_model == "git":
        from src.engines.git import VideoCaptionEngine
        return VideoCaptionEngine()

    elif target_model == "qwen":
        from src.engines.qwen import QwenVideoEngine
        return QwenVideoEngine()
    
    else:
        raise ValueError(f"❌ Unknown model type: '{target_model}'. Supported: blip, git, qwen")