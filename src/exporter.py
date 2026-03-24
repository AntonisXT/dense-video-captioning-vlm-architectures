"""
Exporter Module - Safe File I/O with Error Handling
====================================================

Handles saving of results in multiple formats (CSV, SRT, JSON) with
comprehensive error handling and validation.
"""

import os
import json
import subprocess
import logging
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Custom exception for export-related errors."""
    pass


class Exporter:
    """
    Safe export of dense video captioning results.

    Supports:
    - CSV export (tabular analysis)
    - JSON export (machine-readable results)
    - SRT export (subtitle format)
    - Optional video with burned subtitles (via FFmpeg)
    """

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
        """
        if seconds < 0:
            raise ValueError(f"Timestamp cannot be negative: {seconds}")

        millis = int((seconds - int(seconds)) * 1000)
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)

        return f"{h:02}:{m:02}:{s:02},{millis:03}"

    @staticmethod
    def validate_results(data: List[Dict[str, Any]]) -> None:
        """
        Validate result data structure before export.
        """
        if not data:
            raise ValueError("Cannot export empty results")

        required_fields = ["Scene ID", "Start", "End", "Description"]

        for i, scene in enumerate(data):
            for field in required_fields:
                if field not in scene:
                    raise ValueError(f"Scene {i} missing required field: '{field}'")

            if scene["Start"] >= scene["End"]:
                raise ValueError(
                    f"Scene {i}: Start time ({scene['Start']}) must be < End time ({scene['End']})"
                )

            if scene["Start"] < 0 or scene["End"] < 0:
                raise ValueError(f"Scene {i}: Negative timestamps not allowed")

    @staticmethod
    def save_csv(data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save results as CSV with error handling.
        """
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding="utf-8")
            logger.info(f"✅ CSV saved: {output_path}")

        except PermissionError:
            raise ExportError(f"Permission denied: Cannot write to {output_path}")
        except Exception as e:
            raise ExportError(f"CSV export failed: {e}")

    @staticmethod
    def save_json(
        data: List[Dict[str, Any]],
        video_id: str,
        model_name: str,
        output_path: str,
        mode: str = "e2e",
        raw_scenes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Save results as JSON using a unified, future-proof schema.
        """
        try:
            from src.config import Config
            import platform
            from datetime import datetime, timezone

            def _to_scene(entry: Dict[str, Any]) -> Dict[str, Any]:
                scene_id = int(entry.get("Scene ID", entry.get("scene_id", 0)))
                start = float(entry.get("Start", entry.get("start", 0.0)))
                end = float(entry.get("End", entry.get("end", start)))
                caption = str(entry.get("Description", entry.get("caption", ""))).strip()

                # duration: prefer provided, else compute
                if "Duration" in entry:
                    duration = float(entry["Duration"])
                elif "duration" in entry:
                    duration = float(entry["duration"])
                else:
                    duration = max(0.0, end - start)

                meta = entry.get("meta", {})
                if not isinstance(meta, dict):
                    meta = {}

                return {
                    "scene_id": scene_id,
                    "start": start,
                    "end": end,
                    "duration": float(duration),
                    "caption": caption,
                    "meta": meta,
                }

            scenes_out = [_to_scene(s) for s in data]
            raw_out = [_to_scene(s) for s in raw_scenes] if raw_scenes else None

            output_data: Dict[str, Any] = {
                "schema_version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "video_id": video_id,
                "model": model_name,
                "mode": mode,
                "total_scenes": len(scenes_out),
                "config": Config.to_dict(),  # <-- relies on fixed to_dict()
                "runtime": {
                    "device": str(getattr(Config, "DEVICE", None)),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "scenes": scenes_out,
            }

            if raw_out is not None:
                output_data["raw_scenes"] = raw_out

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            logger.info(f"✅ JSON saved: {output_path}")

        except PermissionError:
            raise ExportError(f"Permission denied: Cannot write to {output_path}")
        except TypeError as e:
            raise ExportError(f"JSON serialization error: {e}")
        except Exception as e:
            raise ExportError(f"JSON export failed: {e}")



    @staticmethod
    def save_srt(data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save results as SRT subtitle file.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in data:
                    scene_id = entry["Scene ID"]
                    start = Exporter.format_timestamp(entry["Start"])
                    end = Exporter.format_timestamp(entry["End"])
                    description = entry["Description"]

                    f.write(f"{scene_id}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{description}\n\n")

            logger.info(f"✅ SRT saved: {output_path}")

        except PermissionError:
            raise ExportError(f"Permission denied: Cannot write to {output_path}")
        except Exception as e:
            raise ExportError(f"SRT export failed: {e}")

    @staticmethod
    def check_ffmpeg() -> bool:
        """
        Check if FFmpeg is available in system PATH.
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def burn_subtitles(
        video_source_path: str,
        srt_path: str,
        output_path: str,
        timeout: int = 600,
    ) -> bool:
        """
        Burn subtitles into video using FFmpeg.
        """
        if not Exporter.check_ffmpeg():
            logger.error("❌ FFmpeg not found in system PATH")
            return False

        if not os.path.exists(video_source_path):
            logger.error(f"❌ Source video not found: {video_source_path}")
            return False

        if not os.path.exists(srt_path):
            logger.error(f"❌ SRT file not found: {srt_path}")
            return False

        # Convert path for FFmpeg (handle Windows paths)
        srt_absolute = os.path.abspath(srt_path)
        srt_path_unix = srt_absolute.replace("\\", "/").replace(":", "\\:")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_source_path,
            "-vf",
            f"subtitles='{srt_path_unix}'",
            "-c:a",
            "copy",
            output_path,
        ]

        try:
            logger.info("🔥 Burning subtitles (this may take a while)...")

            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=True,
            )

            if os.path.exists(output_path):
                logger.info(f"✅ Video with subtitles: {os.path.basename(output_path)}")
                return True

            logger.error("❌ FFmpeg succeeded but output file not found")
            return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ FFmpeg timeout after {timeout}s")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(
                f"❌ FFmpeg error: {e.stderr.decode('utf-8', errors='ignore')}"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during subtitle burning: {e}")
            return False

    @staticmethod
    def save_results(
        data: List[Dict[str, Any]],
        original_filename: str,
        video_source_path: str,
        model_name: str,
        burn_subtitles: bool = False,
        base_results_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save all result formats with comprehensive error handling.
        """
        try:
            Exporter.validate_results(data)
        except ValueError as e:
            raise ExportError(f"Invalid result data: {e}")

        if base_results_dir is None:
            from src.config import Config
            base_results_dir = Config.RESULTS_DIR

        base_name = os.path.splitext(original_filename)[0]
        video_output_dir = os.path.join(base_results_dir, model_name, base_name)

        try:
            os.makedirs(video_output_dir, exist_ok=True)
        except OSError as e:
            raise ExportError(f"Cannot create output directory: {e}")

        csv_path = os.path.join(video_output_dir, f"{base_name}_analysis.csv")
        json_path = os.path.join(video_output_dir, f"{base_name}.json")
        srt_path = os.path.join(video_output_dir, f"{base_name}.srt")
        video_output_path = os.path.join(video_output_dir, f"{base_name}_subtitled.mp4")

        created_files: Dict[str, str] = {}
        errors: List[str] = []

        # CSV
        try:
            Exporter.save_csv(data, csv_path)
            created_files["csv"] = csv_path
        except ExportError as e:
            errors.append(f"CSV: {e}")

        # JSON (critical)
        try:
            Exporter.save_json(data, base_name, model_name, json_path, mode="e2e")
            created_files["json"] = json_path
        except ExportError as e:
            raise ExportError(f"Critical: JSON export failed - {e}")

        # SRT
        try:
            Exporter.save_srt(data, srt_path)
            created_files["srt"] = srt_path
        except ExportError as e:
            errors.append(f"SRT: {e}")

        # Burn subtitles
        if burn_subtitles:
            if "srt" in created_files:
                success = Exporter.burn_subtitles(video_source_path, srt_path, video_output_path)
                if success:
                    created_files["video"] = video_output_path
                else:
                    errors.append("Subtitle burning failed (see logs)")
            else:
                errors.append("Cannot burn subtitles - SRT creation failed")

        logger.info(f"\n💾 Results saved to: {video_output_dir}")
        logger.info(f"   ✅ {len(created_files)} file(s) created successfully")

        if errors:
            logger.warning(f"   ⚠️  {len(errors)} error(s) occurred:")
            for err in errors:
                logger.warning(f"      - {err}")

        return created_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_data = [
        {"Scene ID": 1, "Start": 0.0, "End": 5.5, "Duration": 5.5, "Description": "A person walking down the street"},
        {"Scene ID": 2, "Start": 5.5, "End": 12.3, "Duration": 6.8, "Description": "Cars passing by in traffic"},
    ]

    print("🧪 Testing Exporter module...\n")

    print("Test 1: Timestamp formatting")
    print(f"  0.0 → {Exporter.format_timestamp(0.0)}")
    print(f"  65.5 → {Exporter.format_timestamp(65.5)}")
    print(f"  3723.456 → {Exporter.format_timestamp(3723.456)}")

    print("\nTest 2: Data validation")
    try:
        Exporter.validate_results(test_data)
        print("  ✅ Valid data passed")
    except ValueError as e:
        print(f"  ❌ {e}")

    print("\nTest 3: FFmpeg availability")
    print("  ✅ FFmpeg found" if Exporter.check_ffmpeg() else "  ⚠️  FFmpeg not found")

    print("\n✅ Exporter module tests complete")
