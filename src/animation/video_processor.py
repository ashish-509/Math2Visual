"""
Video Processor

Handles video post-processing operations:
- Format conversion (MP4, GIF, WebM, image sequences)
- Speed adjustment
- Preview generation
"""

import os
import logging
import subprocess
import shutil
import tempfile
from typing import Optional, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video files for format conversion and speed adjustment.
    Uses FFmpeg for video processing operations.
    """
    
    SUPPORTED_FORMATS = ["mp4", "gif", "webm", "png_sequence", "jpg_sequence"]
    QUALITY_PRESETS = {
        "low": {"fps": 15, "scale": 480},
        "medium": {"fps": 30, "scale": 720},
        "high": {"fps": 60, "scale": 1080}
    }
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.gettempdir()
        self.ffmpeg_available = self._check_ffmpeg()
        
        if not self.ffmpeg_available:
            logger.warning("FFmpeg not found. Video processing features will be limited.")
        else:
            logger.info("VideoProcessor initialized with FFmpeg support")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available on the system."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_video_info(self, video_path: str) -> dict:
        """Get information about a video file."""
        if not os.path.exists(video_path):
            return {"error": "File not found"}
        
        if not self.ffmpeg_available:
            # Basic info without FFmpeg
            return {
                "path": video_path,
                "size_bytes": os.path.getsize(video_path),
                "format": os.path.splitext(video_path)[1].lower()
            }
        
        try:
            # Use ffprobe to get video info
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                
                video_stream = None
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_stream = stream
                        break
                
                format_info = info.get("format", {})
                
                return {
                    "path": video_path,
                    "duration": float(format_info.get("duration", 0)),
                    "size_bytes": int(format_info.get("size", 0)),
                    "format": format_info.get("format_name", "unknown"),
                    "width": int(video_stream.get("width", 0)) if video_stream else 0,
                    "height": int(video_stream.get("height", 0)) if video_stream else 0,
                    "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
                    "codec": video_stream.get("codec_name", "unknown") if video_stream else "unknown"
                }
            
            return {"error": "Could not read video info", "path": video_path}
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {"error": str(e), "path": video_path}
    
    def change_speed(self, video_path: str, speed_factor: float, 
                     output_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Change video playback speed.
        
        speed_factor: 0.25 = quarter speed, 2.0 = double speed
                     Valid range: 0.25 to 4.0
        """
        if not os.path.exists(video_path):
            return False, "Input video not found"
        
        if not self.ffmpeg_available:
            return False, "FFmpeg not available for speed adjustment"
        
        # Validate speed factor (0.25x to 4x range)
        if speed_factor < 0.25 or speed_factor > 4.0:
            return False, "Speed factor must be between 0.25 and 4.0"
        
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_speed{speed_factor}x{ext}"
        
        try:
            # FFmpeg filter for speed change
            # For speedup: setpts=0.5*PTS (2x speed)
            # For slowdown: setpts=2*PTS (0.5x speed)
            pts_factor = 1.0 / speed_factor
            
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-filter:v", f"setpts={pts_factor}*PTS",
                "-an",  # Remove audio (will be out of sync anyway)
                output_path
            ]
            
            logger.info(f"Changing speed to {speed_factor}x: {video_path}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Speed changed successfully: {output_path}")
                return True, output_path
            else:
                error = result.stderr or "Unknown error"
                return False, f"FFmpeg error: {error}"
                
        except subprocess.TimeoutExpired:
            return False, "Speed change operation timed out"
        except Exception as e:
            logger.error(f"Error changing speed: {e}")
            return False, str(e)
    
    def convert_to_gif(self, video_path: str, output_path: Optional[str] = None,
                       fps: int = 15, scale: int = 480) -> Tuple[bool, str]:
        """Convert video to animated GIF."""
        if not os.path.exists(video_path):
            return False, "Input video not found"
        
        if not self.ffmpeg_available:
            return False, "FFmpeg not available for GIF conversion"
        
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}.gif"
        
        try:
            # Two-pass GIF creation for better quality
            palette_path = tempfile.mktemp(suffix=".png")
            
            # First pass: generate palette
            cmd_palette = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos,palettegen",
                palette_path
            ]
            
            result = subprocess.run(cmd_palette, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return False, f"Palette generation failed: {result.stderr}"
            
            # Second pass: create GIF with palette
            cmd_gif = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", palette_path,
                "-filter_complex", f"fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse",
                output_path
            ]
            
            result = subprocess.run(cmd_gif, capture_output=True, text=True, timeout=300)
            
            # Cleanup palette
            if os.path.exists(palette_path):
                os.remove(palette_path)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"GIF created: {output_path}")
                return True, output_path
            else:
                return False, f"GIF creation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "GIF conversion timed out"
        except Exception as e:
            logger.error(f"Error converting to GIF: {e}")
            return False, str(e)
    
    def convert_to_webm(self, video_path: str, output_path: Optional[str] = None,
                        quality: str = "medium") -> Tuple[bool, str]:
        """Convert video to WebM format."""
        if not os.path.exists(video_path):
            return False, "Input video not found"
        
        if not self.ffmpeg_available:
            return False, "FFmpeg not available for WebM conversion"
        
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}.webm"
        
        preset = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS["medium"])
        
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-c:v", "libvpx-vp9",
                "-crf", "30",
                "-b:v", "0",
                "-vf", f"scale=-1:{preset['scale']}",
                "-r", str(preset["fps"]),
                output_path
            ]
            
            logger.info(f"Converting to WebM: {video_path}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"WebM created: {output_path}")
                return True, output_path
            else:
                return False, f"WebM conversion failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "WebM conversion timed out"
        except Exception as e:
            logger.error(f"Error converting to WebM: {e}")
            return False, str(e)
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None,
                       format: str = "png", fps: Optional[int] = None) -> Tuple[bool, str]:
        """Extract frames from video as image sequence."""
        if not os.path.exists(video_path):
            return False, "Input video not found"
        
        if not self.ffmpeg_available:
            return False, "FFmpeg not available for frame extraction"
        
        if format not in ["png", "jpg", "jpeg"]:
            return False, "Format must be png or jpg"
        
        if output_dir is None:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(self.output_dir, f"{base}_frames")
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_pattern = os.path.join(output_dir, f"frame_%04d.{format}")
        
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path
            ]
            
            if fps:
                cmd.extend(["-vf", f"fps={fps}"])
            
            cmd.append(output_pattern)
            
            logger.info(f"Extracting frames to: {output_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Count extracted frames
                frames = [f for f in os.listdir(output_dir) if f.endswith(f".{format}")]
                logger.info(f"Extracted {len(frames)} frames to {output_dir}")
                return True, output_dir
            else:
                return False, f"Frame extraction failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Frame extraction timed out"
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return False, str(e)
    
    def create_preview(self, video_path: str, duration: float = 3.0,
                       output_path: Optional[str] = None) -> Tuple[bool, str]:
        """Create a short preview clip from the start of the video."""
        if not os.path.exists(video_path):
            return False, "Input video not found"
        
        if not self.ffmpeg_available:
            return False, "FFmpeg not available for preview creation"
        
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_preview{ext}"
        
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True, output_path
            else:
                return False, f"Preview creation failed: {result.stderr}"
                
        except Exception as e:
            return False, str(e)
    
    def create_thumbnail(self, video_path: str, time_offset: float = 1.0,
                         output_path: Optional[str] = None) -> Tuple[bool, str]:
        """Extract a single frame as thumbnail."""
        if not os.path.exists(video_path):
            return False, "Input video not found"
        
        if not self.ffmpeg_available:
            return False, "FFmpeg not available for thumbnail creation"
        
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}_thumbnail.png"
        
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-ss", str(time_offset),
                "-vframes", "1",
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True, output_path
            else:
                return False, f"Thumbnail creation failed: {result.stderr}"
                
        except Exception as e:
            return False, str(e)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return self.SUPPORTED_FORMATS.copy()
    
    def convert_format(self, video_path: str, target_format: str,
                       output_path: Optional[str] = None,
                       quality: str = "medium") -> Tuple[bool, str]:
        """
        Convert video to specified format.
        Unified interface for all format conversions.
        """
        if target_format not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {target_format}"
        
        if target_format == "gif":
            preset = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS["medium"])
            return self.convert_to_gif(video_path, output_path, 
                                       fps=preset["fps"], scale=preset["scale"])
        
        elif target_format == "webm":
            return self.convert_to_webm(video_path, output_path, quality)
        
        elif target_format in ["png_sequence", "jpg_sequence"]:
            img_format = "png" if target_format == "png_sequence" else "jpg"
            return self.extract_frames(video_path, output_path, format=img_format)
        
        elif target_format == "mp4":
            # Re-encode to standard MP4
            return self._convert_to_mp4(video_path, output_path, quality)
        
        return False, f"Conversion to {target_format} not implemented"
    
    def _convert_to_mp4(self, video_path: str, output_path: Optional[str] = None,
                        quality: str = "medium") -> Tuple[bool, str]:
        """Convert/re-encode video to MP4."""
        if not self.ffmpeg_available:
            return False, "FFmpeg not available"
        
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}_converted.mp4"
        
        preset = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS["medium"])
        
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-vf", f"scale=-1:{preset['scale']}",
                "-r", str(preset["fps"]),
                "-c:a", "aac",
                "-b:a", "128k",
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True, output_path
            else:
                return False, f"MP4 conversion failed: {result.stderr}"
                
        except Exception as e:
            return False, str(e)


# Singleton instance
_video_processor = None


def get_video_processor() -> VideoProcessor:
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
