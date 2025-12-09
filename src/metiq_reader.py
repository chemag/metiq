#!/usr/bin/env python3

"""metiq_reader.py: Unified media reader module.

This module provides a single entry point for all media reading functionality
in metiq. It imports reader implementations from separate modules and provides
factory functions for creating readers.

Available video readers:
- VideoReaderCV2: OpenCV-based reader
- VideoReaderFFmpeg: FFmpeg subprocess-based reader (default)

Available audio readers:
- AudioReaderScipy: Scipy-based reader
- AudioReaderFFmpeg: FFmpeg subprocess-based reader (default)

Usage:
    import metiq_reader

    # Get available readers
    print(metiq_reader.VIDEO_READERS)  # {'cv2': VideoReaderCV2, 'ffmpeg': VideoReaderFFmpeg}
    print(metiq_reader.AUDIO_READERS)  # {'scipy': AudioReaderScipy, 'ffmpeg': AudioReaderFFmpeg}

    # Create a video reader using factory function
    reader = metiq_reader.create_video_reader("input.mp4", reader_type="ffmpeg")

    # Create an audio reader using factory function
    reader = metiq_reader.create_audio_reader("input.mp4", reader_type="ffmpeg", samplerate=16000)
"""

import typing

# Import base classes and dataclasses from generic module
from metiq_reader_generic import (
    VideoMetadata,
    AudioMetadata,
    VideoFrame,
    AudioFrame,
    MediaMetadata,
    VideoReaderBase,
    AudioReaderBase,
    MediaReaderBase,
)

# Import video reader implementations
from metiq_reader_cv2 import (
    VideoReaderCV2,
    HW_DECODER_ENABLE,
)

# Import ffmpeg-based readers
from metiq_reader_ffmpeg import (
    VideoReaderFFmpeg,
    AudioReaderFFmpeg,
    MediaReaderFFmpeg,
)

# Import scipy-based audio reader
from metiq_reader_scipy import (
    AudioReaderScipy,
)


# =============================================================================
# Reader Registry and Factory Functions
# =============================================================================

# Available video readers
VIDEO_READERS = {
    "cv2": VideoReaderCV2,
    "ffmpeg": VideoReaderFFmpeg,
}
DEFAULT_VIDEO_READER = "ffmpeg"

# Available audio readers
AUDIO_READERS = {
    "scipy": AudioReaderScipy,
    "ffmpeg": AudioReaderFFmpeg,
}
DEFAULT_AUDIO_READER = "ffmpeg"


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
