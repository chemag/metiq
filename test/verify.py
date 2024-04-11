#!/usr/bin/env python3

"""Generate and run test content"""
import sys

metiq_path = "../src"
sys.path.append(metiq_path)

import os
import common
import pandas as pd
import argparse
import numpy as np
import inspect
import glob
import re
import verify_unittests
import verify_config as config
from verify_generate import generate_test_file


def run_metiq_cli(**settings):
    filename = settings.get("outfile", "")
    audio_offset = settings.get("audio_offset", 0)
    command = f"python3 {metiq_path}/metiq.py -i {filename} --audio-offset {audio_offset} --calc-all parse --no-cache -d"
    ret, stdout, stderr = common.run(command, debug=config.DEBUG)
    assert ret == 0, f"error: {stderr}"


def verify_metiq_cli(**settings):
    global DEBUG
    failed = False
    doc = settings.get("doc", "")
    filename = settings.get("outfile")
    audio_offset = settings.get("audio_offset", 0.0)
    video_delay = settings.get("video_delay", 0)
    audio_delay = settings.get("audio_delay", 0)
    av_sync = round(video_delay - audio_delay, 2)

    print(f"\n{'-'*20}\n{filename}\n")
    print(f"{doc}")
    print("Audio delay: ", audio_delay)
    print("Video delay: ", video_delay)
    print("A/V sync (calculated): ", av_sync)
    print("Audio offset: ", audio_offset)
    # read the files and compare
    if video_delay > 0:
        videolat = pd.read_csv(f"{filename}.video.latency.csv")
        meanval = videolat["video_latency_sec"].mean()
        result = (
            meanval < config.PREC + video_delay and meanval > video_delay - config.PREC
        )
        if not result:
            failed = True
            print(f"Video delay measurement failed: video delay: {meanval}")

        audiolat = pd.read_csv(f"{filename}.audio.latency.csv")
        meanval = audiolat["audio_latency_sec"].mean()
        result = (
            meanval < config.PREC + audio_delay and meanval > audio_delay - config.PREC
        )
        if not result:
            failed = True
            print(f"Audio delay measurement failed: audio delay: {meanval}")

    avsync = pd.read_csv(f"{filename}.avsync.csv")
    meanval = avsync["av_sync_sec"].mean()
    result = meanval < config.PREC + av_sync and meanval > av_sync - config.PREC
    if not result:
        failed = True
        print(f"Audio/video synchronization measurement failed: a/v sync: {meanval}")

    if failed:
        print(f"{filename}")
        print(f"!!! FAILED\n---\n")
        # Keep files if broken test
        print(f"Video latency:\n{videolat}")
        print(f"Audio latency:\n{audiolat}")
        print(f"A/V sync:\n{avsync}")
        return False
    else:
        print(f"PASS\n{'-'*20}\n")
        # remove all test files
        if not config.KEEP_FILES:
            for file in glob.glob(f"{filename}*"):
                os.remove(file)

    return True


def run_test(**settings):
    settings["doc"] = get_caller_doc()[1]
    generate_test_file(**settings)
    run_metiq_cli(**settings)

    return verify_metiq_cli(**settings)


def get_caller_doc():
    outerframe = inspect.currentframe().f_back.f_back
    name = outerframe.f_code.co_name
    doc = outerframe.f_globals[name].__doc__
    return name, doc


def run_all_tests():
    tests = list_all_tests()
    failed = []
    testcount = len(tests)
    print(f"Total number of tests to run: {testcount}")
    for test in tests:
        result = test[1]()
        if not result:
            print(f"{test[1]} failed, append to failed tests")
            failed.append(test[1])

    passcount = len(tests) - len(failed)
    print(f"Done: {passcount}/{testcount}")

    if len(failed) > 0:
        print("Failed tests:")
        for test in failed:
            print(f"\t{test.__name__}")


def list_all_tests():
    module = sys.modules["verify_unittests"]
    functions = inspect.getmembers(module, inspect.isfunction)
    tests = [
        (int(re.search(r"[0-9]+", str(f))[0]), f)
        for f in functions
        if f[0].startswith("test")
    ]

    tests_sorted = sorted(tests, key=lambda x: int(x[0]))
    return [x[1] for x in tests_sorted]


def print_all_tests():
    tests = list_all_tests()
    for test in tests:
        print(f"{test[0]}: {test[1].__doc__}")


def find_test(testname):
    tests = list_all_tests()
    for test in tests:
        if re.search(f"{testname}$", test[0]):
            return test[1]

    return None


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
        "--keep",
        action="store_true",
        help="keep all files after a run",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list all tests",
    )
    parser.add_argument("--run", type=str, help="Run a specific test number")

    options = parser.parse_args(argv[1:])

    return options


def main(argv):
    # parse options
    options = get_options(argv)

    config.KEEP_FILES = options.keep
    if options.list:
        print_all_tests()
        exit(0)

    if options.run:
        test = find_test(options.run)
        if test:
            test()
        exit(0)
    run_all_tests()


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
