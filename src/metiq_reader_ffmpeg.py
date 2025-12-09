#!/usr/bin/env python3

"""metiq_reader_ffmpeg.py: FFmpeg subprocess-based media reader.

This module provides VideoReaderFFmpeg, AudioReaderFFmpeg, and MediaReaderFFmpeg
classes that read video frames and audio samples from media files using ffmpeg,
providing frame-accurate timestamps via the showinfo filter.

This approach is more deterministic and supports more formats than OpenCV.
"""

import numpy as np
import queue
import re
import subprocess
import threading
import typing

import metiq_reader_generic


class VideoReaderFFmpeg(metiq_reader_generic.VideoReaderBase):
    """Reads video frames from a media file using ffmpeg.

    Uses ffmpeg with the showinfo filter to get per-frame timestamps.
    Outputs raw video frames in the source pixel format to stdout.

    Both stdout (frame data) and stderr (metadata) are read in separate
    threads to avoid blocking.
    """

    # Regex to parse showinfo filter output
    # Example: [Parsed_showinfo_0 @ 0x...] n:   0 pts:      0 pts_time:0 ... s:720x1280 ...
    SHOWINFO_PATTERN = re.compile(
        r"\[Parsed_showinfo_\d+ @ 0x[0-9a-f]+\] "
        r"n:\s*(\d+)\s+"  # frame number
        r"pts:\s*\d+\s+"  # pts (integer)
        r"pts_time:(\S+)\s+"  # pts_time (float)
        r".*?s:(\d+)x(\d+)"  # size WxH
    )

    # Maximum number of frames to buffer
    FRAME_BUFFER_SIZE = 5

    # Pixel format to bytes-per-pixel multiplier (as fraction numerator/denominator)
    # For planar YUV formats, this is the total size divided by width*height
    PIX_FMT_SIZE = {
        # YUV 4:2:0 planar (12 bits per pixel = 1.5 bytes)
        "yuv420p": (3, 2),
        "yuvj420p": (3, 2),
        "yuv420p10le": (3, 1),  # 10-bit = 2 bytes per sample, 1.5x samples
        "yuv420p10be": (3, 1),
        # YUV 4:2:0 semi-planar
        "nv12": (3, 2),
        "nv21": (3, 2),
        # YUV 4:2:2 planar (16 bits per pixel = 2 bytes)
        "yuv422p": (2, 1),
        "yuvj422p": (2, 1),
        "yuv422p10le": (4, 1),
        # YUV 4:4:4 planar (24 bits per pixel = 3 bytes)
        "yuv444p": (3, 1),
        "yuvj444p": (3, 1),
        "yuv444p10le": (6, 1),
        # Packed RGB/BGR (24 bits per pixel = 3 bytes)
        "rgb24": (3, 1),
        "bgr24": (3, 1),
        # Packed RGBA/BGRA (32 bits per pixel = 4 bytes)
        "rgba": (4, 1),
        "bgra": (4, 1),
        "argb": (4, 1),
        "abgr": (4, 1),
        # Grayscale
        "gray": (1, 1),
        "gray16le": (2, 1),
        "gray16be": (2, 1),
        # Packed YUV 4:2:2 (16 bits per pixel = 2 bytes)
        "yuyv422": (2, 1),
        "uyvy422": (2, 1),
    }

    # Default output pixel format - gray for efficiency since video_parse converts to grayscale anyway
    DEFAULT_OUTPUT_PIX_FMT = "gray"

    def __init__(
        self,
        input_file: str,
        pix_fmt: typing.Optional[str] = None,
        count_frames: bool = True,
        debug: int = 0,
        # Additional parameters for compatibility with VideoReaderCV2
        width: int = 0,
        height: int = 0,
        pixel_format: typing.Optional[str] = None,
        threaded: bool = False,
    ):
        """Initialize the video reader.

        Args:
            input_file: Path to the input media file.
            pix_fmt: Output pixel format. If None, uses gray for efficiency.
            count_frames: If True, count frames during probe (slower but accurate).
            debug: Debug level (0=quiet, higher=more verbose).
            width: Ignored (for compatibility with VideoReaderCV2).
            height: Ignored (for compatibility with VideoReaderCV2).
            pixel_format: Ignored (for compatibility with VideoReaderCV2).
            threaded: Ignored (for compatibility with VideoReaderCV2).
        """
        self.input_file = input_file
        # Default to gray for efficiency
        self._target_pix_fmt = pix_fmt if pix_fmt else self.DEFAULT_OUTPUT_PIX_FMT
        self.count_frames = count_frames
        self.debug = debug

        self._process: typing.Optional[subprocess.Popen] = None
        self._stderr_thread: typing.Optional[threading.Thread] = None
        self._stdout_thread: typing.Optional[threading.Thread] = None
        self._metadata_queue: queue.Queue = queue.Queue()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.FRAME_BUFFER_SIZE)
        self._error_output: typing.List[str] = []
        self._width: typing.Optional[int] = None
        self._height: typing.Optional[int] = None
        self._pix_fmt: typing.Optional[str] = None
        self._frame_size: typing.Optional[int] = None
        self._fps: typing.Optional[float] = None
        self._num_frames: int = -1
        self._duration: float = -1.0
        self._started = False
        self._finished = False

    def _calc_frame_size(self, pix_fmt: str, width: int, height: int) -> int:
        """Calculate frame size in bytes for given pixel format and dimensions."""
        if pix_fmt in self.PIX_FMT_SIZE:
            num, den = self.PIX_FMT_SIZE[pix_fmt]
            return (width * height * num) // den
        else:
            # Unknown format - try to query ffmpeg
            # Default to assuming 3 bytes per pixel (like rgb24)
            print(
                f"VideoReader: unknown pixel format '{pix_fmt}', assuming 3 bytes/pixel"
            )
            return width * height * 3

    def start(self) -> bool:
        """Start the ffmpeg process.

        Returns:
            True if the process started successfully, False otherwise.
        """
        if self._started:
            return True

        # First, probe the file to get dimensions, fps, and pixel format
        if not self._probe_video():
            return False

        # Determine output pixel format
        out_pix_fmt = self._target_pix_fmt if self._target_pix_fmt else self._pix_fmt

        # Calculate frame size for output format
        self._frame_size = self._calc_frame_size(out_pix_fmt, self._width, self._height)

        # Build ffmpeg command
        # -vsync passthrough: don't drop or duplicate frames
        # -copyts: preserve original timestamps
        # -vf showinfo: output frame metadata to stderr
        # -f rawvideo: output raw video frames
        # -pix_fmt: output pixel format
        cmd = [
            "ffmpeg",
            "-i",
            self.input_file,
            "-vsync",
            "passthrough",
            "-copyts",
            "-vf",
            "showinfo",
            "-f",
            "rawvideo",
            "-pix_fmt",
            out_pix_fmt,
            "-",
        ]

        if self.debug > 0:
            print(f"VideoReader: running {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            print(f"VideoReader: failed to start ffmpeg: {e}")
            return False

        # Start stderr reader thread (parses showinfo metadata)
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

        # Start stdout reader thread (reads raw frame data)
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True,
        )
        self._stdout_thread.start()

        self._started = True
        return True

    def _probe_video(self) -> bool:
        """Probe the video file to get dimensions, fps, pixel format, and duration."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
        ]
        if self.count_frames:
            cmd.append("-count_frames")
            cmd.extend(
                [
                    "-show_entries",
                    "stream=width,height,r_frame_rate,pix_fmt,nb_read_frames,duration",
                ]
            )
        else:
            cmd.extend(
                ["-show_entries", "stream=width,height,r_frame_rate,pix_fmt,duration"]
            )
        cmd.extend(["-of", "csv=p=0", self.input_file])

        if self.debug > 0:
            print(f"VideoReader: probing with {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"VideoReader: ffprobe failed: {result.stderr}")
                return False

            # Parse output: width,height,pix_fmt,r_frame_rate,duration[,nb_read_frames]
            # Note: ffprobe outputs fields in alphabetical order, not the order specified
            # Note: some fields may be empty (N/A)
            parts = result.stdout.strip().split(",")
            if len(parts) < 4:
                print(f"VideoReader: unexpected ffprobe output: {result.stdout}")
                return False

            self._width = int(parts[0])
            self._height = int(parts[1])

            # Pixel format (3rd field)
            self._pix_fmt = parts[2]

            # Parse frame rate (4th field, could be "30/1" or "30000/1001")
            fps_parts = parts[3].split("/")
            if len(fps_parts) == 2:
                self._fps = float(fps_parts[0]) / float(fps_parts[1])
            else:
                self._fps = float(fps_parts[0])

            # Parse optional duration (5th field)
            self._duration = -1.0
            if len(parts) > 4 and parts[4]:
                try:
                    self._duration = float(parts[4])
                except ValueError:
                    pass

            # Parse optional num_frames (6th field, only present if count_frames is True)
            self._num_frames = -1
            if self.count_frames and len(parts) > 5 and parts[5]:
                try:
                    self._num_frames = int(parts[5])
                except ValueError:
                    pass

            if self.debug > 0:
                print(
                    f"VideoReader: {self._width}x{self._height} @ {self._fps:.2f} fps, "
                    f"pix_fmt={self._pix_fmt}, {self._num_frames} frames, {self._duration:.2f}s"
                )

            return True

        except Exception as e:
            print(f"VideoReader: probe failed: {e}")
            return False

    def _read_stderr(self):
        """Read stderr from ffmpeg process, parsing showinfo output."""
        try:
            for line in self._process.stderr:
                line = line.decode("utf-8", errors="replace").strip()

                if self.debug > 2:
                    print(f"VideoReader stderr: {line}")

                # Try to parse showinfo output
                match = self.SHOWINFO_PATTERN.search(line)
                if match:
                    frame_num = int(match.group(1))
                    pts_time = float(match.group(2))
                    width = int(match.group(3))
                    height = int(match.group(4))

                    self._metadata_queue.put(
                        {
                            "frame_num": frame_num,
                            "pts_time": pts_time,
                            "width": width,
                            "height": height,
                        }
                    )
                else:
                    # Store non-showinfo output for debugging
                    self._error_output.append(line)

        except Exception as e:
            if self.debug > 0:
                print(f"VideoReader: stderr reader error: {e}")
        finally:
            # Signal end of stream
            self._metadata_queue.put(None)

    def _read_stdout(self):
        """Read raw frame data from ffmpeg stdout."""
        try:
            while True:
                # Read exactly frame_size bytes
                data = b""
                remaining = self._frame_size
                while remaining > 0:
                    chunk = self._process.stdout.read(remaining)
                    if not chunk:
                        break
                    data += chunk
                    remaining -= len(chunk)

                if len(data) < self._frame_size:
                    # End of stream or error
                    break

                self._frame_queue.put(data)

        except Exception as e:
            if self.debug > 0:
                print(f"VideoReader: stdout reader error: {e}")
        finally:
            # Signal end of stream
            self._frame_queue.put(None)

    def read(
        self,
    ) -> typing.Tuple[bool, typing.Optional[metiq_reader_generic.VideoFrame]]:
        """Read the next video frame.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and there are no more frames to read.
        """
        if not self._started:
            if not self.start():
                return False, None

        if self._finished:
            return False, None

        # Get frame data from queue
        try:
            data = self._frame_queue.get(timeout=10.0)
            if data is None:
                self._finished = True
                return False, None

            # Convert to numpy array
            frame_data = np.frombuffer(data, dtype=np.uint8)

            # Determine output pixel format
            out_pix_fmt = (
                self._target_pix_fmt if self._target_pix_fmt else self._pix_fmt
            )

            # Reshape frame data based on pixel format for cv2 compatibility
            if out_pix_fmt in ("bgr24", "rgb24"):
                # 3 channels, reshape to (height, width, 3)
                frame_data = frame_data.reshape((self._height, self._width, 3))
            elif out_pix_fmt in ("bgra", "rgba", "argb", "abgr"):
                # 4 channels, reshape to (height, width, 4)
                frame_data = frame_data.reshape((self._height, self._width, 4))
            elif out_pix_fmt == "gray":
                # 1 channel, reshape to (height, width)
                frame_data = frame_data.reshape((self._height, self._width))
            # else: keep as 1D array for other formats (YUV planar, etc.)

        except queue.Empty:
            if self.debug > 0:
                print("VideoReader: timeout waiting for frame data")
            self._finished = True
            return False, None

        # Get metadata from queue
        try:
            metadata = self._metadata_queue.get(timeout=5.0)
            if metadata is None:
                self._finished = True
                return False, None

            frame = metiq_reader_generic.VideoFrame(
                frame_num=metadata["frame_num"],
                pts_time=metadata["pts_time"],
                pix_fmt=out_pix_fmt,
                data=frame_data,
            )
            return True, frame

        except queue.Empty:
            if self.debug > 0:
                print("VideoReader: timeout waiting for frame metadata")
            self._finished = True
            return False, None

    @property
    def width(self) -> int:
        """Video width in pixels."""
        if self._width is None:
            self._probe_video()
        return self._width

    @property
    def height(self) -> int:
        """Video height in pixels."""
        if self._height is None:
            self._probe_video()
        return self._height

    @property
    def fps(self) -> float:
        """Video frame rate."""
        if self._fps is None:
            self._probe_video()
        return self._fps

    @property
    def num_frames(self) -> int:
        """Total number of frames (-1 if unknown)."""
        if self._num_frames == -1 and self._width is None:
            self._probe_video()
        return self._num_frames

    @property
    def duration(self) -> float:
        """Video duration in seconds (-1.0 if unknown)."""
        if self._duration == -1.0 and self._width is None:
            self._probe_video()
        return self._duration

    @property
    def pix_fmt(self) -> str:
        """Source pixel format."""
        if self._pix_fmt is None:
            self._probe_video()
        return self._pix_fmt

    def get_metadata(self) -> typing.Optional[metiq_reader_generic.VideoMetadata]:
        """Get video metadata.

        Returns:
            VideoMetadata or None if probing failed.
        """
        if self._width is None:
            if not self._probe_video():
                return None

        return metiq_reader_generic.VideoMetadata(
            width=self._width,
            height=self._height,
            fps=self._fps,
            pix_fmt=self._pix_fmt,
            num_frames=self._num_frames,
            duration_sec=self._duration,
        )

    def release(self):
        """Release resources."""
        self._finished = True

        if self._process is not None:
            try:
                # Close pipes first to unblock reader threads
                if self._process.stdout:
                    try:
                        self._process.stdout.close()
                    except Exception:
                        pass
                if self._process.stderr:
                    try:
                        self._process.stderr.close()
                    except Exception:
                        pass
                self._process.terminate()
                self._process.wait(timeout=2.0)
            except Exception:
                try:
                    self._process.kill()
                    self._process.wait(timeout=1.0)
                except Exception:
                    pass
            self._process = None

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None

        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1.0)
            self._stdout_thread = None

        self._started = False

    def __del__(self):
        self.release()


class AudioReaderFFmpeg(metiq_reader_generic.AudioReaderBase):
    """Reads audio samples from a media file using ffmpeg.

    By default, uses the native sample rate and channel count from the file.
    Audio is converted to signed 16-bit PCM.
    """

    def __init__(
        self,
        input_file: str,
        samplerate: typing.Optional[int] = None,
        channels: typing.Optional[int] = None,
        debug: int = 0,
    ):
        """Initialize the audio reader.

        Args:
            input_file: Path to the input media file.
            samplerate: Target sample rate. If None, uses native sample rate.
            channels: Target channel count. If None, uses native channel count.
            debug: Debug level (0=quiet, higher=more verbose).
        """
        self.input_file = input_file
        self._target_samplerate = samplerate
        self._target_channels = channels
        self.debug = debug

        self._samplerate: typing.Optional[int] = None
        self._channels: typing.Optional[int] = None
        self._duration: float = -1.0
        self._start_time: float = 0.0  # Audio stream start time offset
        self._samples: typing.Optional[np.ndarray] = None
        self._probed = False
        self._loaded = False

    def _probe_audio(self) -> bool:
        """Probe the audio stream to get native properties."""
        if self._probed:
            return self._samplerate is not None

        self._probed = True

        # Use default output format with explicit key=value pairs instead of CSV,
        # because ffprobe CSV output order is NOT the same as the order specified
        # in -show_entries (it uses its own internal order).
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration,start_time",
            "-of",
            "default=noprint_wrappers=1",
            self.input_file,
        ]

        if self.debug > 0:
            print(f"AudioReader: probing with {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                if self.debug > 0:
                    print(f"AudioReader: ffprobe failed: {result.stderr}")
                return False

            output = result.stdout.strip()
            if not output:
                if self.debug > 0:
                    print(f"AudioReader: no audio stream in {self.input_file}")
                return False

            # Parse output: key=value lines
            values = {}
            for line in output.split("\n"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    values[key] = val

            if "sample_rate" not in values or "channels" not in values:
                print(f"AudioReader: unexpected ffprobe output: {output}")
                return False

            self._samplerate = int(values["sample_rate"])
            self._channels = int(values["channels"])

            if "start_time" in values and values["start_time"]:
                try:
                    self._start_time = float(values["start_time"])
                except ValueError:
                    pass

            if "duration" in values and values["duration"]:
                try:
                    self._duration = float(values["duration"])
                except ValueError:
                    pass

            if self.debug > 0:
                print(
                    f"AudioReader: {self._samplerate} Hz, {self._channels} channels, "
                    f"{self._duration:.2f}s, start_time={self._start_time:.3f}s"
                )

            return True

        except Exception as e:
            print(f"AudioReader: probe failed: {e}")
            return False

    @property
    def samplerate(self) -> int:
        """Audio sample rate in Hz."""
        if self._samplerate is None:
            self._probe_audio()
        # Return target if specified, otherwise native
        if self._target_samplerate is not None:
            return self._target_samplerate
        return self._samplerate if self._samplerate is not None else 0

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        if self._channels is None:
            self._probe_audio()
        # Return target if specified, otherwise native
        if self._target_channels is not None:
            return self._target_channels
        return self._channels if self._channels is not None else 0

    @property
    def start_time(self) -> float:
        """Audio stream start time offset in seconds."""
        if not self._probed:
            self._probe_audio()
        return self._start_time

    def read(self) -> typing.Optional[np.ndarray]:
        """Read all audio samples.

        Returns:
            Numpy array of audio samples (int16), or None if no audio.
            Shape is (num_samples,) for mono, (num_samples, channels) for multi-channel.
        """
        if self._loaded:
            return self._samples

        # Probe first to get native properties
        if not self._probe_audio():
            self._loaded = True
            return None

        # Determine output format
        out_samplerate = (
            self._target_samplerate if self._target_samplerate else self._samplerate
        )
        out_channels = (
            self._target_channels if self._target_channels else self._channels
        )

        # Build ffmpeg command
        # -vn: no video
        # -f s16le: signed 16-bit little-endian PCM
        cmd = [
            "ffmpeg",
            "-i",
            self.input_file,
            "-vn",
            "-ac",
            str(out_channels),
            "-ar",
            str(out_samplerate),
        ]

        # Add audio filter to preserve container timing if there's a start_time offset
        if self._start_time > 0:
            # Calculate delay in samples at INPUT sample rate, not output rate.
            # ffmpeg audio filters (-af) process at the input sample rate, and the
            # -ar option only affects the final output encoding stage. So adelay
            # must use the input 48kHz rate even if we're outputting at 16kHz.
            delay_samples = int(self._start_time * self._samplerate)
            # adelay: prepend silence for start_time offset
            af_filter = f"adelay={delay_samples}S:all=1"
            cmd.extend(["-af", af_filter])
            if self.debug > 0:
                print(
                    f"AudioReader: adding {self._start_time:.3f}s delay "
                    f"({delay_samples} samples @ {self._samplerate}Hz input rate)"
                )

        cmd.extend(["-f", "s16le", "-"])

        if self.debug > 0:
            print(f"AudioReader: running {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                if (
                    "does not contain any stream" in stderr
                    or "no audio" in stderr.lower()
                ):
                    if self.debug > 0:
                        print(f"AudioReader: no audio stream in {self.input_file}")
                else:
                    print(f"AudioReader: ffmpeg error: {stderr[:500]}")
                self._loaded = True
                return None

            # Convert to numpy array
            samples = np.frombuffer(result.stdout, dtype=np.int16)

            # Reshape for multi-channel audio
            if out_channels > 1:
                samples = samples.reshape(-1, out_channels)

            self._samples = samples
            self._loaded = True

            if self.debug > 0:
                num_samples = len(samples) if out_channels == 1 else samples.shape[0]
                duration = num_samples / out_samplerate
                print(f"AudioReader: read {num_samples} samples ({duration:.2f}s)")

            return self._samples

        except Exception as e:
            print(f"AudioReader: read error: {e}")
            self._loaded = True
            return None

    def get_metadata(self) -> typing.Optional[metiq_reader_generic.AudioMetadata]:
        """Get audio metadata.

        Returns:
            AudioMetadata or None if no audio stream.
        """
        if not self._probe_audio():
            return None

        samples = self.read()
        if samples is None:
            return None

        out_samplerate = (
            self._target_samplerate if self._target_samplerate else self._samplerate
        )
        out_channels = (
            self._target_channels if self._target_channels else self._channels
        )
        num_samples = len(samples) if out_channels == 1 else samples.shape[0]

        return metiq_reader_generic.AudioMetadata(
            samplerate=out_samplerate,
            channels=out_channels,
            num_samples=num_samples,
            duration_sec=num_samples / out_samplerate,
        )


class MediaReaderFFmpeg(metiq_reader_generic.MediaReaderBase):
    """Unified media reader for video and audio.

    Provides access to video frames with accurate timestamps and audio samples.
    Uses ffmpeg subprocess for decoding, which is more deterministic and
    supports more formats than OpenCV.

    Example usage:
        reader = MediaReaderFFmpeg("input.mp4")

        # Read video frames
        while True:
            success, frame = reader.read_video_frame()
            if not success:
                break
            # frame.data is raw frame bytes in frame.pix_fmt format
            # frame.pts_time is the presentation timestamp

        # Read audio (all at once)
        audio = reader.read_audio()
    """

    def __init__(
        self,
        input_file: str,
        video_pix_fmt: typing.Optional[str] = None,
        audio_samplerate: typing.Optional[int] = None,
        audio_channels: typing.Optional[int] = None,
        count_frames: bool = True,
        debug: int = 0,
    ):
        """Initialize the media reader.

        Args:
            input_file: Path to the input media file.
            video_pix_fmt: Target video pixel format. If None, uses source format.
            audio_samplerate: Target audio sample rate. If None, uses native.
            audio_channels: Target audio channels. If None, uses native.
            count_frames: If True, count video frames during probe (slower but accurate).
            debug: Debug level (0=quiet, higher=more verbose).
        """
        self.input_file = input_file
        self.debug = debug

        self._video_reader = VideoReaderFFmpeg(
            input_file,
            pix_fmt=video_pix_fmt,
            count_frames=count_frames,
            debug=debug,
        )
        self._audio_reader = AudioReaderFFmpeg(
            input_file,
            samplerate=audio_samplerate,
            channels=audio_channels,
            debug=debug,
        )

    def read_video_frame(
        self,
    ) -> typing.Tuple[bool, typing.Optional[metiq_reader_generic.VideoFrame]]:
        """Read the next video frame.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and there are no more frames to read.
        """
        return self._video_reader.read()

    def read_audio(self) -> typing.Optional[np.ndarray]:
        """Read all audio samples.

        Returns:
            Numpy array of audio samples (int16), or None if no audio.
        """
        return self._audio_reader.read()

    @property
    def video_width(self) -> int:
        """Video width in pixels."""
        return self._video_reader.width

    @property
    def video_height(self) -> int:
        """Video height in pixels."""
        return self._video_reader.height

    @property
    def video_fps(self) -> float:
        """Video frame rate."""
        return self._video_reader.fps

    @property
    def video_num_frames(self) -> int:
        """Total number of video frames (-1 if unknown)."""
        return self._video_reader.num_frames

    @property
    def video_duration(self) -> float:
        """Video duration in seconds (-1.0 if unknown)."""
        return self._video_reader.duration

    @property
    def video_pix_fmt(self) -> str:
        """Video pixel format."""
        return self._video_reader.pix_fmt

    @property
    def audio_samplerate(self) -> int:
        """Audio sample rate in Hz."""
        return self._audio_reader.samplerate

    @property
    def audio_channels(self) -> int:
        """Number of audio channels."""
        return self._audio_reader.channels

    def get_video_metadata(self) -> typing.Optional[metiq_reader_generic.VideoMetadata]:
        """Get video stream metadata."""
        return self._video_reader.get_metadata()

    def get_audio_metadata(self) -> typing.Optional[metiq_reader_generic.AudioMetadata]:
        """Get audio stream metadata."""
        return self._audio_reader.get_metadata()

    def get_metadata(self) -> metiq_reader_generic.MediaMetadata:
        """Get combined metadata for the media file."""
        return metiq_reader_generic.MediaMetadata(
            video=self.get_video_metadata(),
            audio=self.get_audio_metadata(),
        )

    def release(self):
        """Release all resources."""
        self._video_reader.release()

    def __del__(self):
        self.release()
