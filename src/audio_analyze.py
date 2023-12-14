#!/usr/bin/env python3

"""audio_analyze.py module description."""


import argparse
import sys
import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal
import tempfile

import common
import audio_common
from _version import __version__

DEFAULT_MIN_SEPARATION_MSEC = -1
DEFAULT_CORRELATION_FACTOR = 10
DEFAULT_MAX_VALUES = 10000

default_values = {
    "debug": 0,
    "infile": None,
    "outfile": None,
}


def match_buffers(haystack, needle, gain=0, verbose=False):
    """Tries to find needle in haystack using correlation measurement.
    Args:
        haystack: single channel wave haystack
        needle: single channel wave haystack defining the fingerprint to be matched
        gain: additional gain for the haystack before matching
    Returns:
        index: position of the match
        cc: correlation coefficient (0-100)
    """

    global options
    size = len(needle)

    if gain != 0:
        haystack = np.multiply(haystack, gain)
    corr = np.correlate(haystack, needle)
    val = max(corr)

    index = np.where(corr == val)[0][0]

    cc_ = np.corrcoef(haystack[index : index + size], needle)[1, 0] * 100
    if np.isnan(cc_):
        cc_ = 0
    cc = int(cc_ + 0.5)
    if verbose:
        print(f"{val} @ {index}, cc = {cc}")
        visualize_corr(haystack, needle, corr)
    return index, cc


def find_needles(haystack, needle, threshold, samplerate, verbose=False):
    """Searching for needle in haystack in small increments
    Args:
        haystack: single channel wave data
        needle: single channel wave data defining the fingerprint to be matched
        threshold: a threshold for similarity in the range 0 to 100.
        samplerate: the samplerate of the data

    Returns:
        a DataFrame with with the index (a.k.a sample positions), timestamp and correlation value

    Will find places in the audio stream where the original (needle) matches the
    haystack audio stream. It will return a list of tuples with the index, timestamp and
    correlation value. It looks in small chunks of audio and returns the best match
    fullfilling the requirments on absolute threshold and distance to peak in the local
    span of 100ms. The data window to be considered is the length of the needle + 10%
    of the samplerate.
    """

    # Sets how close we can find multiple matches, 100ms
    window = int(0.1 * samplerate)
    max_pos = 0

    silence = np.full((len(needle)), 0)
    haystack = np.append(haystack, silence)
    read_len = int(len(needle) + window)
    ref_duration = len(needle) / samplerate
    counter = 0
    last = 0
    split_times = []
    while last <= len(haystack) - len(needle):
        index, cc = match_buffers(
            haystack[last : last + read_len], needle, verbose=verbose
        )

        index += last
        pos = index - max_pos
        if pos < 0:
            pos = 0
        time = pos / samplerate
        if cc > threshold:
            if (len(split_times) > 0) and (
                abs(time - split_times[-1][1]) < ref_duration / 2
            ):
                if split_times[-1][2] <= cc and time != split_times[-1][1]:
                    split_times.pop(-1)
            else:
                split_times.append([pos, time, cc])

        last += window
        counter += 1

    data = pd.DataFrame()
    labels = ["audio_sample", "timestamp", "correlation"]
    data = pd.DataFrame.from_records(split_times, columns=labels, coerce_float=True)
    return data


def get_correlation_indices(haystack, needle, **kwargs):
    """Find points in haystack with the highest correlation of needle data
    Args:
        haystack: single channel wave data
        needle: single channel wave data defining the fingerprint to be matched

    Returns:
        list of tuples with the index (a.k.a sample positions), timestamp and correlation value

    get_correlation_indices will return a list of indices where the
    the correltion high points are within a certain span between max to
    a min level.
    It can be used to find a number of peak correlation points with
    sufficient separation.
    It looks at the whole file and therefore is unsuitable to find local
    peaks in partials of a file.
    """

    # get optional input parameters
    max_values = kwargs.get("max_values", audio_common.DEFAULT_MAX_VALUES)
    debug = kwargs.get("debug", audio_common.DEFAULT_DEBUG)
    min_separation = kwargs.get("min_separation_samples", -1)
    # calculate the correlation in FP numbers to avoid saturation
    correlation = np.correlate(haystack.astype(np.float32), needle.astype(np.float32))
    # return the <max_values> with the highest correlation values,
    # making sure that all values are separated at least by
    # <min_separation>
    if min_separation <= 0:
        min_separation = len(needle) // 2
    initial_index_list = np.flip(correlation.argsort()[-max_values:])
    index_list = []
    CORRELATION_FACTOR = kwargs.get("correlation_factor", DEFAULT_CORRELATION_FACTOR)
    max_correlation = correlation[initial_index_list[0]]
    min_correlation = max_correlation / CORRELATION_FACTOR
    for index in initial_index_list:
        # index has the next highest correlation
        # 1. ignore index if covered by the previous indices
        found_neighbor = False
        for ii in index_list:
            if found_neighbor:
                break
            if index > (ii - min_separation) and index < (ii + min_separation):
                # too close to a previous index
                if debug > 0:
                    print(f"{index} is too close to {ii}")
                found_neighbor = True
        if found_neighbor:
            continue
        # 2. ignore index if correlation is too low
        if correlation[index] < min_correlation:
            # as indices are sorted by correlation, we can stop here
            break
        index_list.append(index)
    return index_list, correlation


# Returns a list of tuples, where each tuple describes a place in the
# distorted audio stream where the original (reference) audio signal
# has been found. Each tuple consists of 3 elements:
# * (a) `sample_num`: the exact audio sample number (at the reference
#   samplerate, not the source samplerate),
# * (b) `timestamp`: the exact timestamp (calculated from `sample_num`
#   and the samplerate), and
# * (c) `correlation`: the value of the correlation at that timestamp.
def audio_analyze(infile, **kwargs):
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
    # analyze audio file
    audio_results = audio_analyze_wav(wav_filename, **kwargs)
    # sort the index by timestamp
    audio_results = audio_results.sort_values(by=["audio_sample"])
    audio_results = audio_results.reset_index(drop=True)
    return audio_results


def get_audio_duration(infile):
    # open the input
    haystack_samplerate, inaud = scipy.io.wavfile.read(infile)
    audio_duration_samples = len(inaud)
    audio_duration_seconds = audio_duration_samples / haystack_samplerate
    return audio_duration_samples, audio_duration_seconds


def audio_analyze_wav(infile, **kwargs):
    # get optional input parameters
    debug = kwargs.get("debug", audio_common.DEFAULT_DEBUG)
    beep_period_sec = kwargs.get(
        "beep_period_sec", audio_common.DEFAULT_BEEP_PERIOD_SEC
    )
    samplerate = kwargs.get("samplerate", audio_common.DEFAULT_SAMPLERATE)
    min_separation_msec = kwargs.get("min_separation_msec", DEFAULT_MIN_SEPARATION_MSEC)
    min_separation_samples = int(int(min_separation_msec) * int(samplerate) / 1000)
    correlation_factor = kwargs.get("correlation_factor", DEFAULT_CORRELATION_FACTOR)
    echo_analysis = kwargs.get("echo_analysis", False)

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
    beep_duration_samples = kwargs.get(
        "beep_duration_samples", audio_common.DEFAULT_BEEP_DURATION_SAMPLES
    )
    needle_target = audio_common.generate_chirp(beep_period_sec, **kwargs)[
        0:beep_duration_samples
    ]

    audio_results = pd.DataFrame(columns=["audio_sample", "timestamp", "correlation"])

    if echo_analysis:
        # look for indices with more then 20% match (for now)
        audio_results = find_needles(inaud, needle_target, 20, samplerate, False)
    else:
        # calculate the correlation signal
        index_list, correlation = get_correlation_indices(
            inaud,
            needle_target,
            min_separation_samples=min_separation_samples,
            correlation_factor=correlation_factor,
            debug=debug,
        )
        # add a samplerate-based timestamp
        for index in index_list:
            audio_results.loc[len(audio_results.index)] = [
                index,
                index / samplerate,
                int(
                    np.corrcoef(
                        inaud[index : index + len(needle_target)], needle_target
                    )[1, 0]
                    * 100
                ),
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
        "--correlation_factor",
        default=10,
        help="Sets the threshold for triggering hits. Default is a factor 10 between the highest correlation and the lower threshold for triggering hits.",
    )
    parser.add_argument(
        "-e",
        "--echo-analysis",
        dest="echo_analysis",
        action="store_true",
        help="Consider multiple hits in order to calculate time between two consecutive audio trigger points. With this a transmission system can be measured for audio and video latency and auiod/video synchronization.",
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
    audio_results = audio_analyze(
        options.infile,
        debug=options.debug,
        min_separation_msec=options.min_separation_msec,
        min_correlation_factor=options.correlation_factor,
        echo_analysis=options.echo_analysis,
    )
    dump_audio_results(audio_results, options.outfile, options.debug)
    # get audio duration
    audio_duration_samples, audio_duration_seconds = get_audio_duration(options.infile)
    print(f"{audio_duration_samples=}")
    print(f"{audio_duration_seconds=}")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
