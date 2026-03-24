import cv2
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from src.config import Config
from src.utils import clean_caption

class CaptionEngine:
    def __init__(self):
        print(f"🧠 Loading Captioning Model ({Config.BLIP_MODEL_NAME} on {Config.DEVICE})...")
        self.processor = BlipProcessor.from_pretrained(Config.BLIP_MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(
            Config.BLIP_MODEL_NAME, use_safetensors=True
        ).to(Config.DEVICE)

    def _get_frame_metrics(self, cv2_frame):
        """Calculates sharpness (Laplacian variance) and entropy for frame quality assessment."""
        gray = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        
        return sharpness, entropy

    def extract_frames_batch(self, video_path: str, scenes: list) -> list:
            """
            Extracts the keyframe for each scene.
            If Config.ENABLE_KEYFRAME_SELECTION is True: Uses smart selection (Sharpness/Entropy).
            If Config.ENABLE_KEYFRAME_SELECTION is False: Uses the middle frame (Ablation).
            """
            frames = []
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"❌ ERROR: Could not open video {video_path}")
                return [None] * len(scenes)

            # Δυναμικό μήνυμα ανάλογα με το mode
            mode_msg = "Smart (Sharpness/Entropy)" if Config.ENABLE_KEYFRAME_SELECTION else "Baseline (Middle Frame)"
            print(f"   📸 Keyframe Selection Mode: {mode_msg}")
            
            for start_t, end_t in tqdm(scenes, unit="scene", leave=False):
                
                # --- ABLATION LOGIC: MIDDLE FRAME ONLY ---
                if not Config.ENABLE_KEYFRAME_SELECTION:
                    mid_t = (start_t + end_t) / 2
                    cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
                    success, frame = cap.read()
                    
                    if success:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frames.append(img)
                    else:
                        frames.append(None)
                    
                    # Συνεχίζουμε στην επόμενη σκηνή, παρακάμπτοντας τον smart αλγόριθμο
                    continue

                # --- SMART LOGIC: BEST FRAME SELECTION ---
                # (Ο κώδικας από εδώ και κάτω τρέχει μόνο αν ENABLE_KEYFRAME_SELECTION = True)
                duration = end_t - start_t
                
                # Sample timestamps evenly across the scene
                if duration < 0.5:
                    timestamps = [(start_t + end_t) / 2]
                else:
                    ratios = np.linspace(0.1, 0.9, Config.NUM_SAMPLES)
                    timestamps = [start_t + (duration * r) for r in ratios]
                
                candidates = [] 
                for t in timestamps:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    success, frame = cap.read()
                    if success:
                        s, e = self._get_frame_metrics(frame)
                        candidates.append({'frame': frame, 'sharpness': s, 'entropy': e})

                if not candidates:
                    frames.append(None)
                    continue
                    
                # Selection Logic
                if len(candidates) == 1:
                    best_frame = candidates[0]['frame']
                else:
                    max_s = max(c['sharpness'] for c in candidates) + 1e-5
                    max_e = max(c['entropy'] for c in candidates) + 1e-5
                    
                    best_score = -1.0
                    best_frame = candidates[0]['frame']
                    
                    for c in candidates:
                        norm_sharpness = c['sharpness'] / max_s
                        norm_entropy = c['entropy'] / max_e
                        score = (Config.WEIGHT_SHARP * norm_sharpness) + \
                                (Config.WEIGHT_ENTROPY * norm_entropy)
                        
                        if score > best_score:
                            best_score = score
                            best_frame = c['frame']

                img = Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
                frames.append(img)
            
            cap.release()
            return frames

    def generate_caption(self, pil_image) -> str:
        if pil_image is None: return ""
        
        inputs = self.processor(images=pil_image, return_tensors="pt").to(Config.DEVICE)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=Config.BLIP_MAX_LENGTH,
                min_length=Config.BLIP_MIN_LENGTH,
                num_beams=Config.BLIP_NUM_BEAMS,
                repetition_penalty=1.2
            )
            
        raw_caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return clean_caption(raw_caption)