#!/usr/bin/env python3

"""audio_parse.py module description."""


import argparse
import sys
import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal
from scipy.fft import fft, fftfreq
import tempfile
import matplotlib.pyplot as plt

import common
import audio_common
from _version import __version__

DEFAULT_MIN_SEPARATION_MSEC = -1
DEFAULT_MAX_VALUES = 10000
DEFAULT_MIN_MATCH_THRESHOLD = 20

default_values = {
    "debug": 0,
    "infile": None,
    "outfile": None,
    "audio_sample": None,
    "bandpass_filter": False,
}


def bpFilterSignal(data, hp, lp, order, samplerate, verbose):
    """Band pass filter the signal

    Bandpass with butterworth filter with hp and lp

    intputArgs:
        data: input data
        hp: -3dB frequency for high pass
        lp: -3dB frequency for low pass
        samplerate: samplerate of data

    Returns:
        data: filtered version of data
    """

    n = len(data)
    t = 1.0 / samplerate
    if hp is None:
        hp = 0
    if lp is None:
        lp = int(samplerate / 2)
    if verbose:
        fig, (ax1, ax2) = plt.subplots(2)
        yf = fft(data)
        xf = fftfreq(n, t)
        peak = np.argmax(np.abs(yf))
        peakf = int(np.abs(xf[peak]))
        ax1.plot(xf[: int(n / 2)], 1.0 / n * np.abs(yf[: int(n / 2)]))
        ax1.plot(xf[peak], 1.0 / n * np.abs(yf[peak]), "o", color="red")
        ax1.vlines(x=lp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax1.vlines(x=hp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax1.grid()
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("magnitude")
        fig.suptitle(f"FFT non filter vs filtered peak {peakf} Hz")
        print(f"peak: {peakf} Hz")

    sos = scipy.signal.butter(order, [hp, lp], "band", fs=samplerate, output="sos")
    data = scipy.signal.sosfilt(sos, data)
    if verbose:
        yf = fft(data)
        xf = fftfreq(n, t)
        prevPeakf = peakf
        peak = np.argmax(np.abs(yf))
        peakf = int(np.abs(xf[peak]))
        ax2.plot(xf[: int(n / 2)], 1.0 / n * np.abs(yf[: int(n / 2)]))
        ax2.plot(xf[peak], 1.0 / n * np.abs(yf[peak]), "o", color="red")
        ax2.vlines(x=lp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax2.vlines(x=hp, ymax=1.0 / n * max(np.abs(yf)), ymin=0, color="green")
        ax2.grid()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("magnitude")
        print(f"new peak: {peakf} Hz")
        fig.suptitle(
            f"FFT non filter vs filteres, original peak "
            f"{prevPeakf} Hz, filtered {peakf} Hz"
        )
        plt.show()
    return data


def get_correlation_indices(haystack, needle, **kwargs):
    """Find points in haystack with not closer than min_separation_samples and
    above min_match_threshold of needle data
    Args:
        haystack: single channel wave data
        needle: single channel wave data defining the fingerprint to be matched

    Returns:
        list of tuples with the index (a.k.a sample positions) and correlation value

    get_correlation_indices will return a list of indices where the
    the correltion high points are sufficiently separated and the probability of it
    being the needle signal sufficiently high.
    """

    # get optional input parameters
    max_values = kwargs.get("max_values", audio_common.DEFAULT_MAX_VALUES)
    debug = kwargs.get("debug", audio_common.DEFAULT_DEBUG)
    min_separation_samples = int(
        kwargs.get("min_separation_samples", DEFAULT_MIN_SEPARATION_MSEC)
    )
    min_match_threshold = float(
        kwargs.get("min_match_threshold", DEFAULT_MIN_MATCH_THRESHOLD)
    )
    bandpass_filter = kwargs.get("bandpass_filter", False)

    # calculate the correlation in FP numbers to avoid saturation
    needlesize = len(needle)
    # bandpass the input
    # TODO: variable samplerate?
    # TODO: we should have a different definition
    if bandpass_filter:
        chirp_low = audio_common.DEFAULT_BEEP_FREQ
        chirp_high = audio_common.DEFAULT_BEEP_FREQ * 10
        low_freq = int(chirp_low - chirp_low / 2)
        hi_freq = int(chirp_high + chirp_high / 4)
        if debug > 0:
            print(f"{low_freq}= {hi_freq=}")
        haystack = bpFilterSignal(
            haystack, low_freq, hi_freq, 1, audio_common.DEFAULT_SAMPLERATE, debug
        )
    # add a needle length to the front
    haystack = np.concatenate((np.zeros(needlesize), haystack))
    correlation = np.correlate(haystack.astype(np.float32), needle.astype(np.float32))
    # find peaks with a minimum separation of min_separation_samples
    peaks = scipy.signal.find_peaks(correlation, distance=min_separation_samples)[0]
    # calc corrcoeff for filtering
    corrcoeff = [
        np.corrcoef(haystack[x : x + len(needle)], needle)[1, 0] * 100 for x in peaks
    ]
    # filter by threshold
    return [
        [peak - needlesize, cc]
        for peak, cc in zip(peaks, corrcoeff)
        if cc > min_match_threshold
    ]


# Returns a list of tuples, where each tuple describes a place in the
# distorted audio stream where the original (reference) audio signal
# has been found. Each tuple consists of 3 elements:
# * (a) `sample_num`: the exact audio sample number (at the reference
#   samplerate, not the source samplerate),
# * (b) `timestamp`: the exact timestamp (calculated from `sample_num`
#   and the samplerate), and
# * (c) `correlation`: the value of the correlation at that timestamp.
def audio_parse(infile, **kwargs):
    # get optional input parameters
    debug = kwargs.get("debug", audio_common.DEFAULT_DEBUG)
    # convert audio file to wav (mono)
    wav_filename = tempfile.NamedTemporaryFile().name + ".wav"
    # Note that by default this downmixes both channels.
    command = f"ffmpeg -y -i {infile} -vn -ac 1 {wav_filename}"
    ret, stdout, stderr = common.run(command, debug=debug)
    if ret != 0:
        print(f"warn: no audio stream in {infile}")
        return None
    # parse audio file

    audio_results = audio_parse_wav(wav_filename, **kwargs)
    # sort the index by timestamp
    audio_results = audio_results.sort_values(by=["audio_sample"])
    audio_results = audio_results.reset_index(drop=True)
    return audio_results


def get_audio_duration(infile, debug):
    # convert audio file to wav (mono)
    wav_filename = tempfile.NamedTemporaryFile().name + ".wav"
    # Note that by default this downmixes both channels.
    command = f"ffmpeg -y -i {infile} -vn -ac 1 {wav_filename}"
    ret, stdout, stderr = common.run(command, debug=debug)
    if ret != 0:
        print(f"warn: no audio stream in {infile}")
        return None
    # open the input
    haystack_samplerate, inaud = scipy.io.wavfile.read(wav_filename)
    audio_duration_samples = len(inaud)
    audio_duration_seconds = audio_duration_samples / haystack_samplerate
    return audio_duration_samples, audio_duration_seconds


def audio_parse_wav(infile, **kwargs):
    # get optional input parameters
    debug = kwargs.get("debug", audio_common.DEFAULT_DEBUG)
    beep_period_sec = kwargs.get(
        "beep_period_sec", audio_common.DEFAULT_BEEP_PERIOD_SEC
    )
    samplerate = kwargs.get("samplerate", audio_common.DEFAULT_SAMPLERATE)
    min_separation_msec = kwargs.get("min_separation_msec", DEFAULT_MIN_SEPARATION_MSEC)
    min_separation_samples = int(int(min_separation_msec) * int(samplerate) / 1000)
    min_match_threshold = kwargs.get("min_match_threshold")
    audio_sample = kwargs.get("audio_sample", "")
    bandpass_filter = kwargs.get("bandpass_filter", False)
    # open the input
    haystack_samplerate, inaud = scipy.io.wavfile.read(infile)

    # force the input to the experiment samplerate
    if haystack_samplerate != samplerate:
        # need to convert the input's samplerate
        if debug > 0:
            print(
                f"converting {infile} audio from {haystack_samplerate} to {samplerate}"
            )
        # There is a bug (https://github.com/scipy/scipy/issues/15620)
        # resulting in all zeroes unless inout is cast to float
        inaud = scipy.signal.resample_poly(
            inaud.astype(np.float32),
            int(samplerate / 100),
            int(haystack_samplerate / 100),
            padtype="mean",
        )
    # generate a single needle (without the silence)
    needle_target = None
    if audio_sample and len(audio_sample) > 0:
        print(f"read audio sample: {audio_sample}")
        needle_target = scipy.io.wavfile.read(audio_sample)[1]
    else:
        beep_duration_samples = kwargs.get(
            "beep_duration_samples", audio_common.DEFAULT_BEEP_DURATION_SAMPLES
        )
        needle_target = audio_common.generate_chirp(beep_period_sec, **kwargs)[
            0:beep_duration_samples
        ]

    audio_results = pd.DataFrame(columns=["audio_sample", "timestamp", "correlation"])
    # calculate the correlation signal
    if min_separation_samples < 0:
        min_separation_samples = len(needle_target) / 2
    index_list = get_correlation_indices(
        inaud,
        needle_target,
        min_separation_samples=min_separation_samples,
        min_match_threshold=min_match_threshold,
        bandpass_filter=bandpass_filter,
        debug=debug,
    )
    # add a samplerate-based timestamp
    for index, cc in index_list:
        audio_results.loc[len(audio_results.index)] = [
            index,
            index / samplerate,
            cc,
        ]

    if debug > 0:
        print(f"audio_results: {audio_results}")
    return audio_results


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    parser.add_argument(
        "--min_separation_msec",
        default=-1,
        help="Sets a minimal distance between two adjacent signals and sets the shortest detectable time difference in milli seconds. Default is set to halfs the needle length.",
    )
    parser.add_argument(
        "--min_match_threshold",
        default=DEFAULT_MIN_MATCH_THRESHOLD,
        help=f"Sets the minimal correlation coefficient threshold for a signal to be detected. Default is {DEFAULT_MIN_MATCH_THRESHOLD}.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        dest="infile",
        default=default_values["infile"],
        metavar="input-file",
        help="input file [wav]",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file [csv]",
    )
    parser.add_argument(
        "--audio-sample",
        type=str,
        dest="audio_sample",
        default=default_values["audio_sample"],
    )
    parser.add_argument(
        "-bp",
        "--bandpass-filter",
        dest="bandpass_filter",
        action="store_true",
        default=default_values["bandpass_filter"],
        help="Gentle butterworth bandpass filter. Sometimes low correlation hits can improve. Before lowering correlation threshold try filtering.",
    )

    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def dump_audio_results(audio_results, outfile, debug):
    if audio_results is None or len(audio_results) == 0:
        print(f"No audio results: {audio_results = }")
        return
    # write the output as a csv file
    audio_results.to_csv(outfile, index=False)


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get infile/outfile
    if options.infile is None or options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile is None or options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    audio_results = audio_parse(
        options.infile,
        debug=options.debug,
        min_separation_msec=options.min_separation_msec,
        min_match_threshold=options.min_match_threshold,
        audio_sample=options.audio_sample,
        bandpass_filter=options.bandpass_filter,
    )
    dump_audio_results(audio_results, options.outfile, options.debug)
    # get audio duration
    audio_duration_samples, audio_duration_seconds = get_audio_duration(
        options.infile, options.debug
    )
    print(f"{audio_duration_samples=}")
    print(f"{audio_duration_seconds=}")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
