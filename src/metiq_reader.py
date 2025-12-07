#!/usr/bin/env python3

"""metiq_reader.py: Unified media reader module.

This module provides a single entry point for all media reading functionality
in metiq. It consolidates access to different reader implementations and
provides factory functions for creating readers.

Available video readers:
- VideoReaderCV2: OpenCV-based reader (default)
- VideoReaderFFmpeg: FFmpeg subprocess-based reader

Available audio readers:
- AudioReaderScipy: Scipy-based reader (default)
- AudioReaderFFmpeg: FFmpeg subprocess-based reader

Usage:
    import metiq_reader

    # Get available readers
    print(metiq_reader.VIDEO_READERS)  # {'cv2': VideoReaderCV2, 'ffmpeg': VideoReaderFFmpeg}
    print(metiq_reader.AUDIO_READERS)  # {'scipy': AudioReaderScipy, 'ffmpeg': AudioReaderFFmpeg}

    # Create a video reader
    reader = metiq_reader.VideoReaderCV2("input.mp4")

    # Or use factory function
    reader = metiq_reader.create_video_reader("input.mp4", reader_type="cv2")

    # Create an audio reader
    reader = metiq_reader.AudioReaderScipy("input.mp4", samplerate=16000)

    # Or use factory function
    reader = metiq_reader.create_audio_reader("input.mp4", reader_type="scipy", samplerate=16000)
"""

import abc
import cv2
import dataclasses
import numpy as np
import queue
import re
import scipy.io.wavfile
import scipy.signal
import subprocess
import tempfile
import threading
import typing

import common


# =============================================================================
# Generic Interfaces and Dataclasses
# =============================================================================


@dataclasses.dataclass
class VideoMetadata:
    """Metadata about the video stream."""

    width: int
    height: int
    fps: float
    pix_fmt: str
    num_frames: int  # -1 if unknown
    duration_sec: float  # -1.0 if unknown


@dataclasses.dataclass
class AudioMetadata:
    """Metadata about the audio stream."""

    samplerate: int
    channels: int
    num_samples: int  # -1 if unknown
    duration_sec: float  # -1.0 if unknown


@dataclasses.dataclass
class VideoFrame:
    """A single video frame with its metadata."""

    frame_num: int
    pts_time: float  # presentation timestamp in seconds
    pix_fmt: str  # pixel format (e.g., 'yuv420p', 'rgb24', 'gray')
    data: np.ndarray  # raw frame data, shape depends on pix_fmt


@dataclasses.dataclass
class AudioFrame:
    """A chunk of audio samples with its metadata."""

    sample_offset: int  # sample index of first sample in this frame
    pts_time: float  # presentation timestamp in seconds
    samplerate: int
    channels: int
    data: (
        np.ndarray
    )  # audio samples (int16), shape (num_samples,) or (num_samples, channels)


@dataclasses.dataclass
class MediaMetadata:
    """Combined metadata for the media file."""

    video: typing.Optional[VideoMetadata]
    audio: typing.Optional[AudioMetadata]


class VideoReaderBase(abc.ABC):
    """Abstract base class for video readers.

    Implementations must provide:
    - read(): Read the next video frame
    - get_metadata(): Get video stream metadata
    - release(): Release resources

    Properties (width, height, fps, pix_fmt, num_frames, duration) should
    be implemented to return stream information.
    """

    @abc.abstractmethod
    def read(self) -> typing.Tuple[bool, typing.Optional[VideoFrame]]:
        """Read the next video frame.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and there are no more frames to read.
        """
        pass

    @abc.abstractmethod
    def get_metadata(self) -> typing.Optional[VideoMetadata]:
        """Get video stream metadata.

        Returns:
            VideoMetadata or None if no video stream.
        """
        pass

    @abc.abstractmethod
    def release(self):
        """Release resources."""
        pass

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """Video width in pixels."""
        pass

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """Video height in pixels."""
        pass

    @property
    @abc.abstractmethod
    def fps(self) -> float:
        """Video frame rate."""
        pass

    @property
    @abc.abstractmethod
    def pix_fmt(self) -> str:
        """Pixel format."""
        pass

    @property
    @abc.abstractmethod
    def num_frames(self) -> int:
        """Total number of frames (-1 if unknown)."""
        pass

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """Video duration in seconds (-1.0 if unknown)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class AudioReaderBase(abc.ABC):
    """Abstract base class for audio readers.

    Implementations must provide:
    - read(): Read all audio samples
    - get_metadata(): Get audio stream metadata

    Properties (samplerate, channels) should be implemented to return
    stream information.
    """

    @abc.abstractmethod
    def read(self) -> typing.Optional[np.ndarray]:
        """Read all audio samples.

        Returns:
            Numpy array of audio samples (int16), or None if no audio.
            Shape is (num_samples,) for mono, (num_samples, channels) for multi-channel.
        """
        pass

    @abc.abstractmethod
    def get_metadata(self) -> typing.Optional[AudioMetadata]:
        """Get audio stream metadata.

        Returns:
            AudioMetadata or None if no audio stream.
        """
        pass

    @property
    @abc.abstractmethod
    def samplerate(self) -> int:
        """Audio sample rate in Hz."""
        pass

    @property
    @abc.abstractmethod
    def channels(self) -> int:
        """Number of audio channels."""
        pass


class MediaReaderBase(abc.ABC):
    """Abstract base class for unified media readers.

    Provides access to both video and audio streams through a single interface.

    Implementations must provide:
    - read_video_frame(): Read the next video frame
    - read_audio(): Read all audio samples
    - get_video_metadata(): Get video stream metadata
    - get_audio_metadata(): Get audio stream metadata
    - get_metadata(): Get combined metadata
    - release(): Release resources
    """

    @abc.abstractmethod
    def read_video_frame(self) -> typing.Tuple[bool, typing.Optional[VideoFrame]]:
        """Read the next video frame.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and there are no more frames to read.
        """
        pass

    @abc.abstractmethod
    def read_audio(self) -> typing.Optional[np.ndarray]:
        """Read all audio samples.

        Returns:
            Numpy array of audio samples (int16), or None if no audio.
        """
        pass

    @abc.abstractmethod
    def get_video_metadata(self) -> typing.Optional[VideoMetadata]:
        """Get video stream metadata."""
        pass

    @abc.abstractmethod
    def get_audio_metadata(self) -> typing.Optional[AudioMetadata]:
        """Get audio stream metadata."""
        pass

    @abc.abstractmethod
    def get_metadata(self) -> MediaMetadata:
        """Get combined metadata for the media file."""
        pass

    @abc.abstractmethod
    def release(self):
        """Release all resources."""
        pass

    # Video properties
    @property
    @abc.abstractmethod
    def video_width(self) -> int:
        """Video width in pixels."""
        pass

    @property
    @abc.abstractmethod
    def video_height(self) -> int:
        """Video height in pixels."""
        pass

    @property
    @abc.abstractmethod
    def video_fps(self) -> float:
        """Video frame rate."""
        pass

    @property
    @abc.abstractmethod
    def video_pix_fmt(self) -> str:
        """Video pixel format."""
        pass

    @property
    @abc.abstractmethod
    def video_num_frames(self) -> int:
        """Total number of video frames (-1 if unknown)."""
        pass

    @property
    @abc.abstractmethod
    def video_duration(self) -> float:
        """Video duration in seconds (-1.0 if unknown)."""
        pass

    # Audio properties
    @property
    @abc.abstractmethod
    def audio_samplerate(self) -> int:
        """Audio sample rate in Hz."""
        pass

    @property
    @abc.abstractmethod
    def audio_channels(self) -> int:
        """Number of audio channels."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# =============================================================================
# Video Reader Implementations
# =============================================================================

# Global flag for hardware decoder support
HW_DECODER_ENABLE = True


class _VideoCaptureWrapper(cv2.VideoCapture):
    """Threaded wrapper around cv2.VideoCapture for improved performance.

    Uses a background thread to decode frames ahead of time, storing them
    in a queue. This allows the main thread to process frames while the
    decoder thread prepares the next batch.
    """

    def __init__(self, filename, api=0, flags=0):
        if flags:
            super().__init__(filename, api, flags)
        else:
            super().__init__(filename, api)

        self._decode = True
        self._frames = queue.Queue(maxsize=5)
        self._frame_limit = threading.Semaphore(5)
        self._current_time = 0
        self._thread = threading.Thread(target=self._decode_video, daemon=True)
        self._thread.start()

    def read(self):
        """Read the next frame from the queue."""
        if self._decode or self._frames.qsize() > 0:
            try:
                frame, timestamp = self._frames.get(timeout=5.0)
                self._current_time = timestamp
            except queue.Empty:
                return False, None
        else:
            return False, None

        self._frame_limit.release()
        return True, frame

    def get(self, propId):
        """Get a property value."""
        if propId == cv2.CAP_PROP_POS_MSEC:
            return self._current_time
        else:
            return super().get(propId)

    def _decode_video(self):
        """Background thread that decodes frames into the queue."""
        while self._decode:
            ret, frame = super().read()
            current_time = super().get(cv2.CAP_PROP_POS_MSEC)
            if not ret:
                self._decode = False
                break
            self._frame_limit.acquire()
            if self._decode:
                self._frames.put((frame, current_time))

    def release(self):
        """Release resources and stop the decoder thread."""
        self._decode = False
        # Drain the queue
        try:
            while True:
                self._frames.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
        super().release()


class _VideoCaptureYUV:
    """VideoCapture-compatible reader for raw YUV files.

    This class mimics the cv2.VideoCapture interface for reading raw
    YUV planar video files (e.g., yuv420p, nv12, nv21).
    """

    PIX_FMT_COLOR_CONVERSION = {
        "yuv420p": cv2.COLOR_YUV2BGR_I420,
        "nv12": cv2.COLOR_YUV2BGR_NV12,
        "nv21": cv2.COLOR_YUV2BGR_NV21,
    }

    def __init__(self, filename: str, width: int, height: int, pixel_format: str):
        """Initialize the raw YUV reader.

        Args:
            filename: Path to the raw YUV file.
            width: Video width in pixels.
            height: Video height in pixels.
            pixel_format: Pixel format ('yuv420p', 'nv12', or 'nv21').
        """
        self._width = width
        self._height = height
        self._pixel_format = pixel_format
        # Assume 4:2:0 planar (1.5 bytes per pixel)
        self._frame_len = int(width * height * 3 / 2)
        self._shape = (int(height * 1.5), width)
        self._file = open(filename, "rb")
        self._frame_num = 0
        self._fps = 30.0  # Default FPS for raw files

        # Calculate total frames
        self._file.seek(0, 2)  # Seek to end
        file_size = self._file.tell()
        self._file.seek(0)  # Seek back to start
        self._num_frames = file_size // self._frame_len if self._frame_len > 0 else 0

    def isOpened(self) -> bool:
        """Check if the file is open."""
        return self._file is not None and not self._file.closed

    def get(self, propId) -> float:
        """Get a property value."""
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif propId == cv2.CAP_PROP_FPS:
            return self._fps
        elif propId == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._num_frames)
        elif propId == cv2.CAP_PROP_POS_MSEC:
            return (self._frame_num / self._fps) * 1000.0 if self._fps > 0 else 0.0
        return 0.0

    def read(self) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        """Read the next frame and convert to BGR.

        Returns:
            Tuple of (success, bgr_frame).
        """
        ret, yuv = self._read_raw()
        if not ret or yuv is None:
            return False, None

        # Convert YUV to BGR
        color_conv = self.PIX_FMT_COLOR_CONVERSION.get(self._pixel_format)
        if color_conv is None:
            raise ValueError(f"Unsupported pixel format: {self._pixel_format}")

        bgr = cv2.cvtColor(yuv, color_conv)
        self._frame_num += 1
        return True, bgr

    def _read_raw(self) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        """Read raw YUV data for one frame."""
        try:
            raw = self._file.read(self._frame_len)
            if len(raw) < self._frame_len:
                return False, None
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self._shape)
            return True, yuv
        except Exception:
            return False, None

    def release(self):
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.release()


class VideoReaderCV2(VideoReaderBase):
    """Reads video frames from a media file using OpenCV's cv2.VideoCapture.

    This is a wrapper around cv2.VideoCapture that implements the VideoReaderBase
    interface. It supports threaded decoding for improved performance and
    hardware acceleration when available.

    Note: Timestamps from cv2.VideoCapture (CAP_PROP_POS_MSEC) may not be as
    accurate as the ffmpeg showinfo filter. For frame-accurate timestamps,
    use VideoReaderFFmpeg.
    """

    def __init__(
        self,
        input_file: str,
        width: int = 0,
        height: int = 0,
        pixel_format: typing.Optional[str] = None,
        threaded: bool = False,
        debug: int = 0,
    ):
        """Initialize the video reader.

        Args:
            input_file: Path to the input media file.
            width: Expected width (only used for raw YUV files). 0 = auto-detect.
            height: Expected height (only used for raw YUV files). 0 = auto-detect.
            pixel_format: Pixel format for raw YUV files (e.g., 'yuv420p', 'nv12').
                          If None, uses cv2.VideoCapture directly.
            threaded: If True, use threaded decoding for better performance.
            debug: Debug level (0=quiet, higher=more verbose).
        """
        self.input_file = input_file
        self._target_width = width
        self._target_height = height
        self._pixel_format = pixel_format
        self._threaded = threaded
        self.debug = debug

        self._video_capture = None
        self._frame_num = 0
        self._width: typing.Optional[int] = None
        self._height: typing.Optional[int] = None
        self._fps: typing.Optional[float] = None
        self._num_frames: int = -1
        self._duration: float = -1.0
        self._started = False

    def _open_capture(self) -> bool:
        """Open the video capture device."""
        if self._video_capture is not None:
            return True

        try:
            if self._pixel_format is not None:
                # Use raw YUV reader for specific pixel formats
                self._video_capture = _VideoCaptureYUV(
                    self.input_file,
                    self._target_width,
                    self._target_height,
                    self._pixel_format,
                )
            else:
                # Use standard cv2.VideoCapture
                if HW_DECODER_ENABLE:
                    if self._threaded:
                        self._video_capture = _VideoCaptureWrapper(
                            self.input_file,
                            0,
                            (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY),
                        )
                    else:
                        self._video_capture = cv2.VideoCapture(
                            self.input_file,
                            0,
                            (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY),
                        )
                else:
                    if self._threaded:
                        self._video_capture = _VideoCaptureWrapper(self.input_file)
                    else:
                        self._video_capture = cv2.VideoCapture(self.input_file)

            if not self._video_capture.isOpened():
                if self.debug > 0:
                    print(f"VideoReaderCV2: failed to open {self.input_file}")
                return False

            # Get video properties
            self._fps = self._video_capture.get(cv2.CAP_PROP_FPS)
            self._width = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._num_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate duration if num_frames and fps are known
            if self._num_frames > 0 and self._fps > 0:
                self._duration = self._num_frames / self._fps
            else:
                self._duration = -1.0

            # If num_frames is 0 or negative, mark as unknown
            if self._num_frames <= 0:
                self._num_frames = -1

            if self.debug > 0:
                print(
                    f"VideoReaderCV2: {self._width}x{self._height} @ {self._fps:.2f} fps, "
                    f"{self._num_frames} frames, {self._duration:.2f}s"
                )

            self._started = True
            return True

        except Exception as e:
            if self.debug > 0:
                print(f"VideoReaderCV2: error opening {self.input_file}: {e}")
            return False

    def read(self) -> typing.Tuple[bool, typing.Optional[VideoFrame]]:
        """Read the next video frame.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and there are no more frames to read.
        """
        if not self._started:
            if not self._open_capture():
                return False, None

        # Read frame
        status, img = self._video_capture.read()
        if not status:
            return False, None

        # Get timestamp from cv2
        timestamp = self._video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Convert BGR to grayscale if needed, or keep as-is
        # The data is stored as the raw numpy array from cv2
        frame = VideoFrame(
            frame_num=self._frame_num,
            pts_time=timestamp,
            pix_fmt="bgr24",  # cv2.VideoCapture returns BGR format
            data=img,
        )

        self._frame_num += 1
        return True, frame

    def get_metadata(self) -> typing.Optional[VideoMetadata]:
        """Get video metadata.

        Returns:
            VideoMetadata or None if the file cannot be opened.
        """
        if not self._started:
            if not self._open_capture():
                return None

        return VideoMetadata(
            width=self._width,
            height=self._height,
            fps=self._fps,
            pix_fmt="bgr24",  # cv2.VideoCapture returns BGR format
            num_frames=self._num_frames,
            duration_sec=self._duration,
        )

    def release(self):
        """Release resources."""
        if self._video_capture is not None:
            try:
                self._video_capture.release()
            except Exception:
                pass
            self._video_capture = None
        self._started = False

    @property
    def width(self) -> int:
        """Video width in pixels."""
        if self._width is None:
            self._open_capture()
        return self._width if self._width is not None else 0

    @property
    def height(self) -> int:
        """Video height in pixels."""
        if self._height is None:
            self._open_capture()
        return self._height if self._height is not None else 0

    @property
    def fps(self) -> float:
        """Video frame rate."""
        if self._fps is None:
            self._open_capture()
        return self._fps if self._fps is not None else 0.0

    @property
    def pix_fmt(self) -> str:
        """Pixel format (always 'bgr24' for cv2.VideoCapture)."""
        return "bgr24"

    @property
    def num_frames(self) -> int:
        """Total number of frames (-1 if unknown)."""
        if self._num_frames == -1 and not self._started:
            self._open_capture()
        return self._num_frames

    @property
    def duration(self) -> float:
        """Video duration in seconds (-1.0 if unknown)."""
        if self._duration == -1.0 and not self._started:
            self._open_capture()
        return self._duration

    def __del__(self):
        self.release()


class VideoReaderFFmpeg(VideoReaderBase):
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
            pix_fmt: Output pixel format. If None, uses source format.
            count_frames: If True, count frames during probe (slower but accurate).
            debug: Debug level (0=quiet, higher=more verbose).
            width: Ignored (for compatibility with VideoReaderCV2).
            height: Ignored (for compatibility with VideoReaderCV2).
            pixel_format: Alias for pix_fmt (for compatibility).
            threaded: Ignored (for compatibility with VideoReaderCV2).
        """
        self.input_file = input_file
        self._target_pix_fmt = pix_fmt if pix_fmt else pixel_format
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

    def read(self) -> typing.Tuple[bool, typing.Optional[VideoFrame]]:
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

            # Convert to numpy array (keep as 1D byte array)
            frame_data = np.frombuffer(data, dtype=np.uint8)

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

            # Determine output pixel format
            out_pix_fmt = (
                self._target_pix_fmt if self._target_pix_fmt else self._pix_fmt
            )

            frame = VideoFrame(
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

    def get_metadata(self) -> typing.Optional[VideoMetadata]:
        """Get video metadata.

        Returns:
            VideoMetadata or None if probing failed.
        """
        if self._width is None:
            if not self._probe_video():
                return None

        return VideoMetadata(
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
                self._process.terminate()
                self._process.wait(timeout=2.0)
            except Exception:
                try:
                    self._process.kill()
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


# =============================================================================
# Audio Reader Implementations
# =============================================================================


class AudioReaderScipy(AudioReaderBase):
    """Reads audio samples from a media file using scipy.

    Uses ffmpeg to convert the input to WAV format, then scipy.io.wavfile
    to read the samples. Supports resampling to a target sample rate.
    """

    def __init__(
        self,
        input_file: str,
        samplerate: typing.Optional[int] = None,
        channels: int = 1,
        debug: int = 0,
    ):
        """Initialize the audio reader.

        Args:
            input_file: Path to the input media file.
            samplerate: Target sample rate. If None, uses the native sample rate.
            channels: Number of output channels (default: 1 for mono).
            debug: Debug level (0=quiet, higher=more verbose).
        """
        self.input_file = input_file
        self._target_samplerate = samplerate
        self._target_channels = channels
        self.debug = debug

        self._samplerate: typing.Optional[int] = None
        self._channels: typing.Optional[int] = None
        self._samples: typing.Optional[np.ndarray] = None
        self._loaded = False
        self._wav_file: typing.Optional[str] = None

    def _convert_to_wav(self) -> typing.Optional[str]:
        """Convert input file to WAV format using ffmpeg.

        Returns:
            Path to the temporary WAV file, or None on failure.
        """
        if self._wav_file is not None:
            return self._wav_file

        # Create a temporary WAV file
        self._wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        # Build ffmpeg command
        # -vn: no video
        # -ac: number of audio channels
        cmd = f"ffmpeg -y -i {self.input_file} -vn -ac {self._target_channels} {self._wav_file}"

        if self.debug > 0:
            print(f"AudioReaderScipy: running {cmd}")

        ret, stdout, stderr = common.run(cmd, debug=self.debug)
        if ret != 0:
            if self.debug > 0:
                print(f"AudioReaderScipy: no audio stream in {self.input_file}")
            self._wav_file = None
            return None

        return self._wav_file

    def read(self) -> typing.Optional[np.ndarray]:
        """Read all audio samples.

        Returns:
            Numpy array of audio samples (int16), or None if no audio.
            Shape is (num_samples,) for mono, (num_samples, channels) for multi-channel.
        """
        if self._loaded:
            return self._samples

        self._loaded = True

        # Convert to WAV
        wav_file = self._convert_to_wav()
        if wav_file is None:
            return None

        try:
            # Read WAV file
            native_samplerate, samples = scipy.io.wavfile.read(wav_file)
            self._samplerate = native_samplerate
            self._channels = self._target_channels

            if self.debug > 0:
                print(
                    f"AudioReaderScipy: read {len(samples)} samples at {native_samplerate} Hz"
                )

            # Resample if needed
            if (
                self._target_samplerate is not None
                and native_samplerate != self._target_samplerate
            ):
                if self.debug > 0:
                    print(
                        f"AudioReaderScipy: resampling from {native_samplerate} to {self._target_samplerate}"
                    )
                # There is a bug (https://github.com/scipy/scipy/issues/15620)
                # resulting in all zeroes unless input is cast to float
                samples = scipy.signal.resample_poly(
                    samples.astype(np.float32),
                    int(self._target_samplerate / 100),
                    int(native_samplerate / 100),
                    padtype="mean",
                )
                # Convert back to int16
                samples = samples.astype(np.int16)
                self._samplerate = self._target_samplerate

            self._samples = samples
            return self._samples

        except Exception as e:
            if self.debug > 0:
                print(f"AudioReaderScipy: read error: {e}")
            return None

    def get_metadata(self) -> typing.Optional[AudioMetadata]:
        """Get audio metadata.

        Returns:
            AudioMetadata or None if no audio stream.
        """
        # Ensure audio is loaded to get metadata
        samples = self.read()
        if samples is None:
            return None

        num_samples = len(samples) if self._channels == 1 else samples.shape[0]

        return AudioMetadata(
            samplerate=self._samplerate,
            channels=self._channels,
            num_samples=num_samples,
            duration_sec=num_samples / self._samplerate,
        )

    @property
    def samplerate(self) -> int:
        """Audio sample rate in Hz."""
        if self._samplerate is None:
            self.read()
        return self._samplerate if self._samplerate is not None else 0

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels if self._channels is not None else self._target_channels


class AudioReaderFFmpeg(AudioReaderBase):
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
        self._samples: typing.Optional[np.ndarray] = None
        self._probed = False
        self._loaded = False

    def _probe_audio(self) -> bool:
        """Probe the audio stream to get native properties."""
        if self._probed:
            return self._samplerate is not None

        self._probed = True

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration",
            "-of",
            "csv=p=0",
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

            # Parse output: sample_rate,channels,duration
            parts = output.split(",")
            if len(parts) < 2:
                print(f"AudioReader: unexpected ffprobe output: {output}")
                return False

            self._samplerate = int(parts[0])
            self._channels = int(parts[1])

            if len(parts) > 2 and parts[2]:
                try:
                    self._duration = float(parts[2])
                except ValueError:
                    pass

            if self.debug > 0:
                print(
                    f"AudioReader: {self._samplerate} Hz, {self._channels} channels, "
                    f"{self._duration:.2f}s"
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
            "-f",
            "s16le",
            "-",
        ]

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

    def get_metadata(self) -> typing.Optional[AudioMetadata]:
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

        return AudioMetadata(
            samplerate=out_samplerate,
            channels=out_channels,
            num_samples=num_samples,
            duration_sec=num_samples / out_samplerate,
        )


# =============================================================================
# Unified Media Reader
# =============================================================================


class MediaReaderFFmpeg(MediaReaderBase):
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

    def read_video_frame(self) -> typing.Tuple[bool, typing.Optional[VideoFrame]]:
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

    def get_video_metadata(self) -> typing.Optional[VideoMetadata]:
        """Get video stream metadata."""
        return self._video_reader.get_metadata()

    def get_audio_metadata(self) -> typing.Optional[AudioMetadata]:
        """Get audio stream metadata."""
        return self._audio_reader.get_metadata()

    def get_metadata(self) -> MediaMetadata:
        """Get combined metadata for the media file."""
        return MediaMetadata(
            video=self.get_video_metadata(),
            audio=self.get_audio_metadata(),
        )

    def release(self):
        """Release all resources."""
        self._video_reader.release()

    def __del__(self):
        self.release()


# =============================================================================
# Reader Registry and Factory Functions
# =============================================================================

# Available video readers
VIDEO_READERS = {
    "cv2": VideoReaderCV2,
    "ffmpeg": VideoReaderFFmpeg,
}
DEFAULT_VIDEO_READER = "cv2"

# Available audio readers
AUDIO_READERS = {
    "scipy": AudioReaderScipy,
    "ffmpeg": AudioReaderFFmpeg,
}
DEFAULT_AUDIO_READER = "scipy"


def create_video_reader(
    input_file: str,
    reader_type: typing.Optional[str] = None,
    reader_class: typing.Optional[typing.Type[VideoReaderBase]] = None,
    **kwargs,
) -> VideoReaderBase:
    """Create a video reader of the specified type.

    Args:
        input_file: Path to the input media file.
        reader_type: Reader type ('cv2' or 'ffmpeg'). If None, uses default.
        reader_class: Reader class to use directly. If provided, overrides reader_type.
        **kwargs: Additional arguments passed to the reader constructor.

    Returns:
        A VideoReaderBase instance.

    Raises:
        ValueError: If reader_type is not recognized.
    """
    # If a class is provided directly, use it
    if reader_class is not None:
        return reader_class(input_file, **kwargs)

    if reader_type is None:
        reader_type = DEFAULT_VIDEO_READER

    if reader_type not in VIDEO_READERS:
        raise ValueError(
            f"Unknown video reader type: {reader_type}. "
            f"Available: {list(VIDEO_READERS.keys())}"
        )

    return VIDEO_READERS[reader_type](input_file, **kwargs)


def create_audio_reader(
    input_file: str,
    reader_type: typing.Optional[str] = None,
    reader_class: typing.Optional[typing.Type[AudioReaderBase]] = None,
    **kwargs,
) -> AudioReaderBase:
    """Create an audio reader of the specified type.

    Args:
        input_file: Path to the input media file.
        reader_type: Reader type ('scipy' or 'ffmpeg'). If None, uses default.
        reader_class: Reader class to use directly. If provided, overrides reader_type.
        **kwargs: Additional arguments passed to the reader constructor.

    Returns:
        An AudioReaderBase instance.

    Raises:
        ValueError: If reader_type is not recognized.
    """
    # If a class is provided directly, use it
    if reader_class is not None:
        return reader_class(input_file, **kwargs)

    if reader_type is None:
        reader_type = DEFAULT_AUDIO_READER

    if reader_type not in AUDIO_READERS:
        raise ValueError(
            f"Unknown audio reader type: {reader_type}. "
            f"Available: {list(AUDIO_READERS.keys())}"
        )

    return AUDIO_READERS[reader_type](input_file, **kwargs)


def list_video_readers() -> typing.List[str]:
    """Return list of available video reader types."""
    return list(VIDEO_READERS.keys())


def list_audio_readers() -> typing.List[str]:
    """Return list of available audio reader types."""
    return list(AUDIO_READERS.keys())
