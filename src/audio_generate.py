#!/usr/bin/env python3

"""gaudio_gen.py module description."""


import scipy.io.wavfile

import audio_common


def audio_generate(duration_sec, output_filename, **kwargs):
    aud = audio_common.generate_beep(duration_sec, **kwargs)
    scipy.io.wavfile.write(output_filename, audio_common.DEFAULT_SAMPLERATE, aud)
