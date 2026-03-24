import cv2
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from src.config import Config
from src.utils import clean_caption

class VideoCaptionEngine:
    """
    Engine for Microsoft GIT (Generative Image-to-text Transformer).
    Processes multiple frames per scene to generate spatiotemporal captions.
    """
    def __init__(self):
        self.model_name = Config.GIT_MODEL_NAME
        print(f"🧠 Loading Video Model ({self.model_name})...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(Config.DEVICE)
        
        # GIT standard input size (typically 6 frames)
        self.num_frames = 6

    def extract_frames_batch(self, video_path: str, scenes: list) -> list:
        video_clips = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return [None] * len(scenes)

        print(f"   🎞️  Extracting Video Clips ({self.num_frames} frames/scene)...")
        
        for start_t, end_t in tqdm(scenes, unit="scene", leave=False):
            # Sample evenly spaced timestamps
            timestamps = np.linspace(start_t, end_t, self.num_frames + 2)[1:-1]
            
            clip_frames = []
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                success, frame = cap.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip_frames.append(frame)
            
            # --- PADDING & SLICING ---
            if not clip_frames:
                video_clips.append(None)
            else:
                # Pad with last frame if insufficient
                while len(clip_frames) < self.num_frames:
                    clip_frames.append(clip_frames[-1])
                
                # Truncate if excess
                video_clips.append(clip_frames[:self.num_frames])
        
        cap.release()
        return video_clips

    def generate_caption(self, frame_list) -> str:
        if not frame_list: 
            return ""
        
        try:
            inputs = self.processor(images=frame_list, return_tensors="pt").to(Config.DEVICE)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs.pixel_values, 
                    max_length=Config.GIT_MAX_LENGTH,
                    min_length=Config.GIT_MIN_LENGTH,  
                )
                
            raw_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return clean_caption(raw_caption)
            
        except Exception as e:
            # Prevent single-scene failure from crashing the pipeline
            print(f"⚠️ GIT Generation Error: {str(e)}")
            return ""