#!/usr/bin/env python3

"""gaudio_gen.py module description."""


import scipy.io.wavfile

import audio_common


def audio_generate(duration_sec, output_filename, **kwargs):
    """Generate audio file with given duration and parameters."""
    audio_sample = kwargs.get("audio_sample", "")
    aud = None
    if len(audio_sample) > 0:
        aud = audio_common.generate_sample_based(duration_sec, **kwargs)
    else:
        aud = audio_common.generate_chirp(duration_sec, **kwargs)
    scipy.io.wavfile.write(output_filename, audio_common.DEFAULT_SAMPLERATE, aud)
