#!/usr/bin/env python3

"""gaudio_common.py module description."""


import numpy as np
import scipy
import scipy.signal as signal

DEFAULT_SAMPLERATE = 16000
DEFAULT_SCALE = 0.7
# DEFAULT_SCALE = 1
DEFAULT_DEBUG = 0
DEFAULT_PRE_SAMPLES = 0
DEFAULT_BEEP_FREQ = 440
DEFAULT_BEEP_DURATION_SAMPLES = 4800
DEFAULT_BEEP_PERIOD_SEC = 3

DEFAULT_MAX_VALUES = 10000

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html
def generate_chirp(duration_sec, **kwargs):
    # get optional input parameters
    pre_samples = kwargs.get("pre_samples", DEFAULT_PRE_SAMPLES)
    samplerate = kwargs.get("samplerate", DEFAULT_SAMPLERATE)
    freq = kwargs.get("freq", DEFAULT_BEEP_FREQ)
    duration_samples = kwargs.get("duration_samples", DEFAULT_BEEP_DURATION_SAMPLES)
    period_sec = kwargs.get("period_sec", DEFAULT_BEEP_PERIOD_SEC)
    scale = kwargs.get("scale", DEFAULT_SCALE)
    # generate a <duration_samples> chirp signal from freq * 3x freq
    t = np.linspace(0, duration_samples / samplerate, duration_samples)
    array = signal.chirp(
        t, f0=freq, f1=freq * 3, t1=duration_samples / samplerate, method="logarithmic"
    )
    array = (array * (2**15) * scale).astype(np.int16)
    ramp = np.arange(0, 1, 0.01)
    array[:100] = array[:100] * ramp
    array[-100:] = array[-100:] * ramp[::-1]
    # compose the final signal of beeps and silences
    aud = np.zeros(pre_samples, dtype=np.int16)
    cur_sample = 0
    total_samples = duration_sec * samplerate
    while cur_sample < total_samples:
        # add a beep
        aud = np.append(aud, array)
        cur_sample += duration_samples
        if cur_sample >= total_samples:
            break
        # add silence
        num_samples_silence = int(
            min(
                (samplerate * period_sec) - duration_samples,
                total_samples - cur_sample,
            )
        )
        zero_array = np.zeros(num_samples_silence, dtype=np.int16)
        aud = np.append(aud, zero_array)
        cur_sample += num_samples_silence
    return aud


# https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html
def generate_sin(num_samples, freq, samplerate, scale):
    t = np.arange(num_samples)
    samples = (2**15) * scale * np.sin(2 * np.pi * t * freq / samplerate)
    # make sure casting works
    samples[samples > np.iinfo(np.int16).max] = np.iinfo(np.int16).max
    samples[samples < np.iinfo(np.int16).min] = np.iinfo(np.int16).min
    return samples.astype(np.int16)


def generate_beep(duration_sec, **kwargs):
    # get optional input parameters
    pre_samples = kwargs.get("pre_samples", DEFAULT_PRE_SAMPLES)
    samplerate = kwargs.get("samplerate", DEFAULT_SAMPLERATE)
    beep_freq = kwargs.get("beep_freq", DEFAULT_BEEP_FREQ)
    beep_duration_samples = kwargs.get(
        "beep_duration_samples", DEFAULT_BEEP_DURATION_SAMPLES
    )
    beep_period_sec = kwargs.get("beep_period_sec", DEFAULT_BEEP_PERIOD_SEC)
    scale = kwargs.get("scale", DEFAULT_SCALE)
    # generate a <beep_duration_samples> sin signal
    beep_array = generate_sin(beep_duration_samples, beep_freq, samplerate, scale=scale)
    # compose the final signal of beeps and silences
    aud = np.zeros(pre_samples, dtype=np.int16)
    cur_sample = 0
    total_samples = duration_sec * samplerate
    while cur_sample < total_samples:
        # add a beep
        aud = np.append(aud, beep_array)
        cur_sample += beep_duration_samples
        if cur_sample >= total_samples:
            break
        # add silence
        num_samples_silence = int(
            min(
                (samplerate * beep_period_sec) - beep_duration_samples,
                total_samples - cur_sample,
            )
        )
        zero_array = np.zeros(num_samples_silence, dtype=np.int16)
        aud = np.append(aud, zero_array)
        cur_sample += num_samples_silence
    return aud


def main():
    aud = generate_chirp(
        1,
        pre_samples=0,
        samplerate=16000,
        beep_freq=440,
        beep_duration_samples=4800,
        beep_period_sec=3,
        scale=0.7,
    )
    scipy.io.wavfile.write("chirp_16k.wav", 16000, aud)


if __name__ == "__main__":
    main()
