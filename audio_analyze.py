#!/usr/bin/env python3

"""audio_analyze.py module description."""


import argparse
import sys
import numpy as np
import scipy.io.wavfile
import scipy.signal
import tempfile

import common
import audio_common
from _version import __version__


default_values = {
    "debug": 0,
    "infile": None,
    "outfile": None,
}


def get_correlation_indices(haystack, needle, **kwargs):
    # get optional input parameters
    max_values = kwargs.get("max_values", audio_common.DEFAULT_MAX_VALUES)
    debug = kwargs.get("debug", audio_common.DEFAULT_DEBUG)
    # calculate the correlation in FP numbers to avoid saturation
    correlation = np.correlate(haystack.astype(np.float32), needle.astype(np.float32))
    # return the <max_values> with the highest correlation values,
    # making sure that all values are separated at least by
    # <min_separation>
    min_separation = len(needle) // 2
    initial_index_list = np.flip(correlation.argsort()[-max_values:])
    index_list = []
    CORRELATION_FACTOR = 10
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
    assert ret == 0, f"error: {stderr}"
    # analyze audio file
    audio_results = audio_analyze_wav(wav_filename, **kwargs)
    # sort the index list
    audio_results.sort()
    return audio_results


def audio_analyze_wav(infile, **kwargs):
    # get optional input parameters
    debug = kwargs.get("debug", audio_common.DEFAULT_DEBUG)
    beep_period_sec = kwargs.get(
        "beep_period_sec", audio_common.DEFAULT_BEEP_PERIOD_SEC
    )
    samplerate = kwargs.get("samplerate", audio_common.DEFAULT_SAMPLERATE)
    # open the input
    haystack_samplerate, inaud = scipy.io.wavfile.read(infile)
    # force the input to the experiment samplerate
    if haystack_samplerate != samplerate:
        if debug > 0:
            print(
                f"converting {infile} audio from {haystack_samplerate} to {samplerate}"
            )
        inaud = scipy.signal.decimate(inaud, int(haystack_samplerate / samplerate))
    # generate a 1-period needle
    needle = audio_common.generate_beep(beep_period_sec, **kwargs)
    # calculate the correlation signal
    index_list, correlation = get_correlation_indices(inaud, needle)
    # add a samplerate-based timestamp
    audio_results = [
        (index, index / samplerate, correlation[index]) for index in index_list
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
        "infile",
        type=str,
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def dump_results(audio_results, outfile, debug):
    # write the output as a csv file
    with open(outfile, "w") as fd:
        fd.write("index,timestamp,correlation\n")
        for index, timestamp, correlation in audio_results:
            fd.write(f"{index},{timestamp},{correlation}\n")


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
        options.debug,
    )
    dump_results(audio_results, options.outfile, options.debug)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
