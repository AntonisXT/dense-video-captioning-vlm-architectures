"""
Main Entry Point - Dense Video Captioning System
=================================================
"""

import os
import time
import argparse
import sys
from typing import Optional, List, Dict, Any
from colorama import Fore, Style, init as colorama_init

# Initialize colorama
colorama_init(autoreset=True)

# Modules
from src.config import Config
from src.logger import log
from src.scene_engine import SceneEngine
from src.model_factory import get_caption_engine
from src.merger import SceneMerger
from src.exporter import Exporter, ExportError


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dense Video Captioning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python main.py --model blip
            python main.py --model git --burn n --limit 10
            python main.py --model qwen --no-merging
        """
    )
    
    # Model selection
    parser.add_argument("--model", type=str, default=Config.CAPTION_MODEL_TYPE, choices=["blip", "git", "qwen"], help="Caption model to use")
    
    # Subtitle burning
    parser.add_argument("--burn", type=str, choices=['y', 'n'], default=None, help="Burn subtitles: 'y' for YES, 'n' for NO")
    
    # Processing limit
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process")
    
    # Ablation study flags
    parser.add_argument("--no-merging", action="store_true", help="Disable semantic scene merging")   
    parser.add_argument("--random-frames", action="store_true", help="Use random frame selection")
    parser.add_argument("--ablation-name", type=str, default=None, help="Custom experiment name")
    
    # Resume mode
    parser.add_argument("--force", action="store_true", help="Force reprocess completed videos")
    
    return parser.parse_args()


def apply_ablation_settings(args: argparse.Namespace) -> None:
    """Overrides Config settings based on ablation flags."""
    if args.no_merging:
        Config.ENABLE_SCENE_MERGING = False
        log.info("🔬 Ablation: Scene merging DISABLED")    
    
    if args.random_frames:
        Config.ENABLE_KEYFRAME_SELECTION = False
        log.info("🔬 Ablation: Random frame selection ENABLED")
    
    if args.ablation_name:
        Config.ABLATION_EXPERIMENT = args.ablation_name
        log.info(f"🔬 Ablation experiment: {args.ablation_name}")


def get_video_files(videos_dir: str) -> List[str]:
    """Retrieves supported video files from the directory."""
    if not os.path.exists(videos_dir):
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    all_files = os.listdir(videos_dir)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    
    if not video_files:
        raise ValueError(f"No video files found in {videos_dir}")
    
    return video_files


def is_video_processed(filename: str, model_name: str, force: bool = False) -> bool:
    """Checks if video processing artifacts already exist."""
    if force: return False
    
    filename_no_ext = os.path.splitext(filename)[0]
    expected_output_dir = os.path.join(Config.RESULTS_DIR, model_name, filename_no_ext)
    
    if os.path.exists(expected_output_dir):
        json_path = os.path.join(expected_output_dir, f"{filename_no_ext}.json")
        if os.path.exists(json_path):
            return True
    
    return False


def process_single_video(video_path: str, filename: str, caption_engine, scene_merger: SceneMerger, burn_subs: bool) -> bool:
    """Executes the full processing pipeline for a single video."""

    try:
        # Step 1: Scene Detection (Added \n)
        log.info(Fore.MAGENTA + "\n📍 Step 1/4: Scene Detection")
        scene_engine = SceneEngine(video_path)
        scenes = scene_engine.detect_scenes()
        
        valid_scenes = [s for s in scenes if (s[1] - s[0]) >= 0.2]
        log.info(f"   ✅ Found {len(scenes)} scenes ({len(valid_scenes)} valid)")
        
        if not valid_scenes:
            log.warning(f"   ⚠️  No valid scenes found for {filename}")
            return False
        
        # Step 2: Frame Extraction (Added \n)
        log.info(Fore.MAGENTA + "\n📍 Step 2/4: Keyframe Extraction")
        try:
            visual_inputs = caption_engine.extract_frames_batch(video_path, valid_scenes)
        except Exception as e:
            log.error(f"   ❌ Frame extraction failed: {e}", exc_info=True)
            return False
        
        # Step 3: Caption Generation (Added \n)
        log.info(Fore.MAGENTA + f"\n📍 Step 3/4: Caption Generation ({Config.CAPTION_MODEL_TYPE.upper()})")
        raw_results = []
        
        for i, ((start, end), visual_input) in enumerate(zip(valid_scenes, visual_inputs)):
            if visual_input is None:
                log.warning(f"   ⚠️  Scene {i+1}: Frame extraction failed")
                continue
            
            try:
                description = caption_engine.generate_caption(visual_input)
                
                if description:
                    log.info(f"   📝 Scene {i+1}/{len(valid_scenes)}: {description}")
                    raw_results.append({
                        "Scene ID": i + 1,
                        "Start": start,
                        "End": end,
                        "Duration": round(end - start, 2),
                        "Description": description
                    })
                else:
                    log.warning(f"   ⚠️  Scene {i+1}: Empty caption generated")
                    
            except Exception as e:
                log.error(f"   ❌ Scene {i+1}: Caption generation failed - {e}")
                continue
        
        if not raw_results:
            log.error(f"   ❌ No captions generated for {filename}")
            return False
        
        log.info(f"   ✅ Generated {len(raw_results)} captions")
        
        # Step 4: Scene Merging (Added \n)
        log.info(Fore.MAGENTA + "\n📍 Step 4/4: Semantic Scene Merging")
        
        if Config.ENABLE_SCENE_MERGING:
            try:
                final_results = scene_merger.merge_scenes(raw_results)
                log.info(f"   ✅ Merged to {len(final_results)} final scenes")
            except Exception as e:
                log.error(f"   ⚠️  Merging failed, using raw results: {e}")
                final_results = raw_results
        else:
            log.info("   ℹ️  Scene merging disabled (ablation mode)")
            final_results = raw_results
        
        # Step 5: Export Results (Added \n)
        log.info(Fore.MAGENTA + "\n📍 Step 5/5: Exporting Results")
        try:
            created_files = Exporter.save_results(
                final_results, filename, video_path, Config.CAPTION_MODEL_TYPE, burn_subs
            )
            log.info(f"   ✅ Saved {len(created_files)} output file(s)")
            return True
            
        except ExportError as e:
            log.error(f"   ❌ Export failed: {e}")
            return False
    
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log.error(f"   ❌ Unexpected error in {filename}: {e}", exc_info=True)
        return False


def main() -> int:
    """Main execution flow"""
    try:
        args = parse_arguments()
        
        log.info("\n" + "="*60)
        log.info("🚀 DENSE VIDEO CAPTIONING SYSTEM")
        log.info("="*60)
        
        try:
            Config.initialize_directories()
            Config.validate()
        except (ValueError, RuntimeError) as e:
            log.error(f"❌ Configuration Error:\n{e}")
            return 1
        
        # Apply Configuration
        Config.CAPTION_MODEL_TYPE = args.model
        apply_ablation_settings(args)
        Config.print_summary()
        
        # Subtitle Logic
        if args.burn == 'y':
            burn_subs = True
            log.info("🔥 Subtitle burning: ENABLED")
        elif args.burn == 'n':
            burn_subs = False
            log.info("❄️  Subtitle burning: DISABLED")
        else:
            print(Fore.WHITE + "\nDo you want to create videos with burned-in subtitles?")
            user_choice = input(Fore.YELLOW + "Press 'y' for YES or 'n' for NO: ").strip().lower()
            burn_subs = (user_choice == 'y')
            log.info(f"{'🔥' if burn_subs else '❄️ '} Subtitle burning: {'ENABLED' if burn_subs else 'DISABLED'}")
        
        # Locate Videos
        try:
            video_files = get_video_files(Config.VIDEOS_DIR)
            log.info(f"📁 Found {len(video_files)} video(s) in {Config.VIDEOS_DIR}")
        except (FileNotFoundError, ValueError) as e:
            log.error(f"❌ {e}")
            return 1
        
        # Initialize Models
        log.info(f"\n{'='*60}")
        log.info("⚙️  INITIALIZING MODELS")
        log.info(f"{'='*60}")
        
        try:
            caption_engine = get_caption_engine()
            scene_merger = SceneMerger()
        except Exception as e:
            log.error(f"❌ Model initialization failed: {e}", exc_info=True)
            return 1
        
        # Processing Loop
        start_time = time.time()
        stats = {'processed': 0, 'success': 0, 'skipped': 0, 'failed': 0}
        
        log.info(f"\n{'='*60}")
        log.info("📹 PROCESSING VIDEOS")
        log.info(f"{'='*60}\n")
        
        for i, video_file in enumerate(video_files, 1):
            if args.limit is not None and stats['processed'] >= args.limit:
                log.info(f"🛑 Reached limit of {args.limit} video(s). Stopping.")
                break
            
            if is_video_processed(video_file, Config.CAPTION_MODEL_TYPE, args.force):
                log.info(f"⏩ [{i}/{len(video_files)}] SKIP: {video_file} (already processed)")
                stats['skipped'] += 1
                continue
            
            log.info(f"\n{'='*60}")
            log.info(f"▶️  [{i}/{len(video_files)}] Processing: {video_file}")
            log.info(f"{'='*60}")

            success = process_single_video(
                os.path.join(Config.VIDEOS_DIR, video_file),
                video_file, caption_engine, scene_merger, burn_subs
            )
            
            stats['processed'] += 1
            if success:
                stats['success'] += 1
                log.info(f"✅ [{i}/{len(video_files)}] SUCCESS: {video_file}\n")
            else:
                stats['failed'] += 1
                log.error(f"❌ [{i}/{len(video_files)}] FAILED: {video_file}\n")
        
        # Final Summary
        total_time = time.time() - start_time
        log.info(f"\n{'='*60}")
        log.info("📊 PROCESSING SUMMARY")
        log.info(f"{'='*60}")
        log.info(f"   Total Videos:      {len(video_files)}")
        log.info(f"   ✅ Successful:     {stats['success']}")
        log.info(f"   ⏩ Skipped:        {stats['skipped']}")
        log.info(f"   ❌ Failed:         {stats['failed']}")
        log.info(f"   ⏱️  Total Time:     {total_time:.1f}s")
        
        if stats['processed'] > 0:
            log.info(f"   ⚡ Avg Time:       {total_time/stats['processed']:.1f}s per video")
        
        log.info(f"{'='*60}\n")
        
        if stats['success'] > 0:
            log.info(f"✅ COMPLETED! Results saved in: {Config.RESULTS_DIR}/{Config.CAPTION_MODEL_TYPE}/")
        
        return 0 if stats['failed'] == 0 else 1
    
    except KeyboardInterrupt:
        log.warning("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        return 130
    
    except Exception as e:
        log.error(f"\n❌ CRITICAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())