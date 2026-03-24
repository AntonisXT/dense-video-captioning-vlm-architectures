import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from src.config import Config
from src.utils import clean_caption

class QwenVideoEngine:
    """
    Engine for Qwen2-VL (Visual Language Model).
    Uses 4-bit quantization to fit in consumer GPUs.
    """
    def __init__(self):
        self.model_name = Config.QWEN_MODEL_NAME
        print(f"🧠 Loading Qwen2-VL ({self.model_name}) in 4-bit mode...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        try:
            attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation=attn_impl
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                min_pixels=256*28*28, 
                max_pixels=1280*28*28
            )
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load Qwen2-VL. {e}")
            raise e
            
        # Strategy for limited VRAM (e.g., 4GB-8GB): Use 5 frames
        self.num_frames = 5

    def extract_frames_batch(self, video_path: str, scenes: list) -> list:
        video_clips = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return [None] * len(scenes)

        print(f"   🎞️  Sampling {self.num_frames} frames per scene (Qwen Strategy)...")
        
        for start_t, end_t in tqdm(scenes, unit="scene", leave=False):
            timestamps = np.linspace(start_t, end_t, self.num_frames + 2)[1:-1]
            
            clip_frames = []
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                success, frame = cap.read()
                if success:
                    # Qwen expects PIL Images
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip_frames.append(Image.fromarray(frame_rgb))
            
            if not clip_frames:
                video_clips.append(None)
            else:
                while len(clip_frames) < self.num_frames:
                    clip_frames.append(clip_frames[-1])
                video_clips.append(clip_frames[:self.num_frames])
        
        cap.release()
        return video_clips

    def generate_caption(self, frame_list) -> str:
        if not frame_list: 
            return ""
        
        try:
            # Construct the VLM prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frame_list,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": "Describe this video scene in a single, concise sentence. Focus objectively on the main subject(s), the specific action or motion occurring, and the visible environment or background. Include key visual details like colors, objects, or movements only if they are prominent."},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.QWEN_MAX_LENGTH,
                    do_sample=Config.QWEN_DO_SAMPLE,
                    temperature=Config.QWEN_TEMPERATURE,
                    top_p=Config.QWEN_TOP_P
                )
            
            # Decode only the new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return clean_caption(output_text)

        except Exception as e:
            if "out of memory" in str(e).lower():
                print("⚠️ GPU OOM: Qwen exhausted available VRAM. Clearing cache.")
                torch.cuda.empty_cache()
            else:
                print(f"⚠️ Qwen Generation Error: {e}")
            return ""