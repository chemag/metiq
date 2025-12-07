#!/usr/bin/env python3

"""metiq_reader_scipy.py: Scipy-based audio reader.

This module provides AudioReaderScipy, an audio reader implementation using
scipy.io.wavfile. It implements the AudioReaderBase interface defined in
metiq_reader_generic.py.

The reader uses ffmpeg to convert the input media file to a WAV file,
then uses scipy to read the audio samples.

Note: This reader only supports audio streams. For video, use the cv2 or
ffmpeg-based readers from metiq_reader_cv2.py or metiq_reader_ffmpeg.py.
"""

import numpy as np
import scipy.io.wavfile
import scipy.signal
import tempfile
import typing

import common
import metiq_reader_generic


class AudioReaderScipy(metiq_reader_generic.AudioReaderBase):
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

    def get_metadata(self) -> typing.Optional[metiq_reader_generic.AudioMetadata]:
        """Get audio metadata.

        Returns:
            AudioMetadata or None if no audio stream.
        """
        # Ensure audio is loaded to get metadata
        samples = self.read()
        if samples is None:
            return None

        num_samples = len(samples) if self._channels == 1 else samples.shape[0]

        return metiq_reader_generic.AudioMetadata(
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
