#!/usr/bin/env python3

"""metiq_reader_cv2.py: OpenCV-based video reader.

This module provides VideoReaderCV2, a video reader implementation using
OpenCV's cv2.VideoCapture. It implements the VideoReaderBase interface
defined in metiq_reader_generic.py.

Note: This reader only supports video streams. For audio, use the ffmpeg-based
reader from metiq_reader_ffmpeg.py.
"""

import cv2
import numpy as np
import queue
import threading
import typing

import metiq_reader_generic


# Global flag for hardware decoder support
HW_DECODER_ENABLE = True


class VideoReaderCV2(metiq_reader_generic.VideoReaderBase):
    """Reads video frames from a media file using OpenCV's cv2.VideoCapture.

    This is a wrapper around cv2.VideoCapture that implements the VideoReaderBase
    interface. It supports threaded decoding for improved performance and
    hardware acceleration when available.

    Note: Timestamps from cv2.VideoCapture (CAP_PROP_POS_MSEC) may not be as
    accurate as the ffmpeg showinfo filter. For frame-accurate timestamps,
    use VideoReaderFFmpeg from metiq_reader_ffmpeg.py.
    """

    def __init__(
        self,
        input_file: str,
        width: int = 0,
        height: int = 0,
        pixel_format: typing.Optional[str] = None,
        pix_fmt: typing.Optional[str] = None,
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
            pix_fmt: Output pixel format ('gray' or 'bgr24'). If 'gray', converts
                     BGR output to grayscale. Default is 'bgr24'.
            threaded: If True, use threaded decoding for better performance.
            debug: Debug level (0=quiet, higher=more verbose).
        """
        self.input_file = input_file
        self._target_width = width
        self._target_height = height
        self._pixel_format = pixel_format
        self._output_pix_fmt = pix_fmt if pix_fmt else "bgr24"
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

    def read(
        self,
    ) -> typing.Tuple[bool, typing.Optional[metiq_reader_generic.VideoFrame]]:
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

        # Convert to requested output format
        if self._output_pix_fmt == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        frame = metiq_reader_generic.VideoFrame(
            frame_num=self._frame_num,
            pts_time=timestamp,
            pix_fmt=self._output_pix_fmt,
            data=img,
        )

        self._frame_num += 1
        return True, frame

    def get_metadata(self) -> typing.Optional[metiq_reader_generic.VideoMetadata]:
        """Get video metadata.

        Returns:
            VideoMetadata or None if the file cannot be opened.
        """
        if not self._started:
            if not self._open_capture():
                return None

        return metiq_reader_generic.VideoMetadata(
            width=self._width,
            height=self._height,
            fps=self._fps,
            pix_fmt=self._output_pix_fmt,
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
        """Pixel format."""
        return self._output_pix_fmt

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
