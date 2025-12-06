#!/usr/bin/env python3

"""metiq_reader_generic.py: Generic media reader interfaces and dataclasses.

This module provides:
- Dataclasses for video/audio metadata and frames
- Abstract base classes for VideoReader, AudioReader, and MediaReader

Concrete implementations are in:
- metiq_reader_ffmpeg.py: ffmpeg subprocess-based reader
- metiq_reader_cv2.py: OpenCV-based reader
"""

import abc
import dataclasses
import numpy as np
from typing import Optional, Tuple


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

    video: Optional[VideoMetadata]
    audio: Optional[AudioMetadata]


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
    def read(self) -> Tuple[bool, Optional[VideoFrame]]:
        """Read the next video frame.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and there are no more frames to read.
        """
        pass

    @abc.abstractmethod
    def get_metadata(self) -> Optional[VideoMetadata]:
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
    def read(self) -> Optional[np.ndarray]:
        """Read all audio samples.

        Returns:
            Numpy array of audio samples (int16), or None if no audio.
            Shape is (num_samples,) for mono, (num_samples, channels) for multi-channel.
        """
        pass

    @abc.abstractmethod
    def get_metadata(self) -> Optional[AudioMetadata]:
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
    def read_video_frame(self) -> Tuple[bool, Optional[VideoFrame]]:
        """Read the next video frame.

        Returns:
            Tuple of (success, frame). If success is False, frame is None
            and there are no more frames to read.
        """
        pass

    @abc.abstractmethod
    def read_audio(self) -> Optional[np.ndarray]:
        """Read all audio samples.

        Returns:
            Numpy array of audio samples (int16), or None if no audio.
        """
        pass

    @abc.abstractmethod
    def get_video_metadata(self) -> Optional[VideoMetadata]:
        """Get video stream metadata."""
        pass

    @abc.abstractmethod
    def get_audio_metadata(self) -> Optional[AudioMetadata]:
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
