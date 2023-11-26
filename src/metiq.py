#!/usr/bin/env python

"""metiq.py module description."""


import argparse
import math
import numpy as np
import pandas as pd
import os
import sys
import tempfile

import audio_common
import audio_generate
import audio_analyze
import vft
import video_common
import video_generate
import video_analyze
import common


from _version import __version__

FUNC_CHOICES = {
    "help": "show help options",
    "generate": "generate reference video",
    "analyze": "analyze distorted video",
}

DEFAULT_NUM_FRAMES = 1800

default_values = {
    "debug": 0,
    # vft parameters
    "vft_id": vft.DEFAULT_VFT_ID,
    "vft_tag_border_size": vft.DEFAULT_TAG_BORDER_SIZE,
    "luma_threshold": vft.DEFAULT_LUMA_THRESHOLD,
    # video parameters
    "num_frames": DEFAULT_NUM_FRAMES,
    "fps": video_common.DEFAULT_FPS,
    "width": video_common.DEFAULT_WIDTH,
    "height": video_common.DEFAULT_HEIGHT,
    "pixel_format": video_common.DEFAULT_PIXEL_FORMAT,
    # audio parameters
    "pre_samples": audio_common.DEFAULT_PRE_SAMPLES,
    "samplerate": audio_common.DEFAULT_SAMPLERATE,
    "beep_freq": audio_common.DEFAULT_BEEP_FREQ,
    "beep_duration_samples": audio_common.DEFAULT_BEEP_DURATION_SAMPLES,
    "beep_period_sec": audio_common.DEFAULT_BEEP_PERIOD_SEC,
    "scale": audio_common.DEFAULT_SCALE,
    "min_separation_msec": audio_analyze.DEFAULT_MIN_SEPARATION_MSEC,
    "correlation_factor": audio_analyze.DEFAULT_CORRELATION_FACTOR,
    # common parameters
    "func": "help",
    "infile": None,
    "outfile": None,
    "audio_offset": 0,
    "lock_layout": False,
}


def media_file_generate(
    width,
    height,
    fps,
    num_frames,
    vft_id,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    scale,
    outfile,
    debug,
):
    # calculate the frame period
    beep_period_frames = beep_period_sec * fps
    vft_layout = vft.VFTLayout(width, height, vft_id)
    max_frame_num = 2**vft_layout.numbits
    frame_period = beep_period_frames * (max_frame_num // beep_period_frames)
    # generate the (raw) video input
    video_filename = tempfile.NamedTemporaryFile().name + ".rgb24"
    rem = f"period: {beep_period_sec} freq_hz: {beep_freq} samples: {beep_duration_samples}"
    video_generate.video_generate(
        width,
        height,
        fps,
        num_frames,
        frame_period,
        video_filename,
        "default",
        vft_id,
        rem,
        debug,
    )
    duration_sec = num_frames / fps
    # generate the (raw) audio input
    audio_filename = tempfile.NamedTemporaryFile().name + ".wav"
    audio_generate.audio_generate(
        duration_sec,
        audio_filename,
        pre_samples=pre_samples,
        samplerate=samplerate,
        beep_freq=beep_freq,
        beep_duration_samples=beep_duration_samples,
        beep_period_sec=beep_period_sec,
        scale=scale,
        debug=debug,
    )
    # put them together
    command = "ffmpeg -y "
    command += f"-f rawvideo -pixel_format rgb24 -s {width}x{height} -r {fps} -i {video_filename} "
    command += f"-i {audio_filename} "
    command += f"-c:v libx264 -pix_fmt yuv420p -c:a aac {outfile}"
    ret, stdout, stderr = common.run(command, debug=debug)
    assert ret == 0, f"error: {stderr}"
    # clean up raw files
    os.remove(video_filename)
    os.remove(audio_filename)


def estimate_audio_frame_num(
    prev_video_frame_num, video_frame_num, prev_video_ts, video_ts, audio_ts
):
    audio_frame_num = prev_video_frame_num + (
        video_frame_num - prev_video_frame_num
    ) * ((audio_ts - prev_video_ts) / (video_ts - prev_video_ts))
    return audio_frame_num


def estimate_avsync(video_results, fps, audio_results, beep_period_sec, debug=0):
    video_index = 0
    # get the video frame_num corresponding to each audio timestamp
    audio_frame_num_list = []
    for _audio_sample, audio_ts, _audio_correlation in audio_results:
        if video_index >= len(video_results):
            break
        # find the video matches whose timestamps surround the audio one
        while True:
            # find a video entry that has a valid reading
            try:
                prev_video_ts, _, prev_video_frame_num = video_results[video_index][1:4]
            except:
                print(f"error: {video_results[video_index] =}")
                continue
            if prev_video_frame_num is not None:
                break
            video_index += 1
        video_index += 1
        get_next_audio_ts = False
        while not get_next_audio_ts and video_index < len(video_results):
            video_ts, _, video_frame_num = video_results[video_index][1:4]
            if video_frame_num is None:
                video_index += 1
                continue
            if audio_ts >= prev_video_ts and audio_ts <= video_ts:
                # found a match
                audio_frame_num = estimate_audio_frame_num(
                    prev_video_frame_num,
                    video_frame_num,
                    prev_video_ts,
                    video_ts,
                    audio_ts,
                )
                if debug > 0:
                    print(
                        f"estimate_avsync: found match"
                        f" prev_video {{ frame_num: {prev_video_frame_num} ts: {prev_video_ts} }}"
                        f" this_video {{ frame_num: {video_frame_num} ts: {video_ts} }}"
                        f" audio {{ ts: {audio_ts} estimated_frame_num: {audio_frame_num} }}"
                    )
                audio_frame_num_list.append(audio_frame_num)
                get_next_audio_ts = True
            elif audio_ts < prev_video_ts:
                # could not find the audio timestamp
                get_next_audio_ts = True
            prev_video_ts, prev_video_frame_num = video_ts, video_frame_num
            # linearize the video frame number
            video_index += 1
    # get the closest video_frame_num
    video_frame_num_period = fps * beep_period_sec
    avsync_sec_list = []
    for audio_frame_num in audio_frame_num_list:
        if video_frame_num_period > 0 and not np.isnan(audio_frame_num):
            candidate_1 = (
                math.floor(audio_frame_num / video_frame_num_period)
                * video_frame_num_period
            )
            candidate_2 = (
                math.ceil(audio_frame_num / video_frame_num_period)
                * video_frame_num_period
            )
            if (audio_frame_num - candidate_1) < (candidate_2 - audio_frame_num):
                video_frame_num = candidate_1
            else:
                video_frame_num = candidate_2
            # Following ITU-R BT.1359-1 rec here: "positive value indicates that
            # sound is advanced with respect to vision"
            # It can also be though of as a delay time on video i.e.
            # a negative delay means that the audio is leading.
            avsync_sec = (video_frame_num - audio_frame_num) / fps
            avsync_sec_list.append(
                [video_frame_num, round(audio_frame_num / fps, 3), avsync_sec]
            )
    return avsync_sec_list


def media_file_analyze(
    width,
    height,
    fps,
    num_frames,
    pixel_format,
    luma_threshold,
    pre_samples,
    samplerate,
    beep_freq,
    beep_duration_samples,
    beep_period_sec,
    scale,
    infile,
    outfile,
    debug,
    **kwargs,
):
    echo_analysis = kwargs.get("echo_analysis", False)
    audio_offset = kwargs.get("audio_offset", 0)
    lock_layout = kwargs.get("lock_layout", False)

    # 1. analyze the video stream
    video_results = None
    video_delta_info = None
    avsync_sec_list = None
    errors = None
    if debug > 0:
        path = f"{infile}.video.csv"
        if os.path.exists(f"{infile}.video.csv"):
            pvideo_results = pd.read_csv(f"{infile}.video.csv")
            video_results = (
                pvideo_results[
                    [
                        "frame_num",
                        "timestamp",
                        "expected_frame_num",
                        "video_frame_num_read",
                        "delta",
                    ]
                ]
                .to_records(index=False)
                .tolist()
            )
        if os.path.exists(f"{infile}.video_delta_info.csv"):
            vdi = pd.read_csv(f"{infile}.video_delta_info.csv")
            video_delta_info = vdi.set_index("0").T.to_dict("list")

        if os.path.exists(f"{infile}.video.errors.csv"):
            perrors = pd.read_csv(f"{infile}.video.errors.csv")
            errors = (
                perrors[["frame", "timestamp", "exception"]]
                .to_records(index=False)
                .tolist()
            )
    if video_results == None:
        video_results, video_delta_info, errors = video_analyze.video_analyze(
            infile,
            width,
            height,
            fps,
            pixel_format,
            luma_threshold,
            lock_layout=lock_layout,
            debug=debug,
        )
        perrors = pd.DataFrame(errors, columns=["frame", "timestamp", "exception"])
        if perrors is not None and len(perrors) > 0:
            perrors.to_csv(f"{infile}.video.errors.csv")
        if debug > 0:
            if video_delta_info is not None and len(video_delta_info) > 0:
                pvideo_delta_info = pd.DataFrame(video_delta_info.items())
                pvideo_delta_info.to_csv(f"{infile}.video_delta_info.csv", index=False)
                pvideo_results = pd.DataFrame(
                    video_results,
                    columns=[
                        "frame_num",
                        "timestamp",
                        "expected_frame_num",
                        "video_frame_num_read",
                        "delta",
                    ],
                )
                pvideo_results.to_csv(path)

    # 2. analyze the audio stream
    audio_results = audio_analyze.audio_analyze(
        infile,
        pre_samples=pre_samples,
        samplerate=samplerate,
        beep_freq=beep_freq,
        beep_duration_samples=beep_duration_samples,
        beep_period_sec=beep_period_sec,
        scale=scale,
        min_separation_msec=kwargs.get("min_separation_msec", 50),
        correlation_factor=kwargs.get("correlation_factor", 10),
        echo_analysis=echo_analysis,
        debug=debug,
    )
    if debug > 0:
        paudio_results = pd.DataFrame(
            audio_results, columns=["audio_sample", "timestamp", "correlation"]
        )
        paudio_results.to_csv(f"{infile}.audio.csv")
    if debug > 1:
        print(f"{audio_results = }")
    if not echo_analysis:
        # 3a. estimate a/v sync
        avsync_sec_list = estimate_avsync(
            video_results, fps, audio_results, beep_period_sec, debug
        )
        if debug > 0:
            pav_sync_list = pd.DataFrame(
                avsync_sec_list, columns=["video_frame_num", "audio_sec", "avsync_sec"]
            )
            pav_sync_list.to_csv(f"{infile}.avsync.csv")

        # 4. dump results to file
        dump_results(video_results, video_delta_info, audio_results, outfile, debug)

        return video_delta_info, avsync_sec_list

    else:
        latencies = []
        # 3b. calculate audio latency, video latency and the difference between the two
        audio_latencies, video_latencies, avsyncs, combined = calculate_latency(
            audio_results, video_results, beep_period_sec, audio_offset, debug
        )
        if debug > 0:
            audio_latencies.to_csv(
                f"{os.path.splitext(outfile)[0]}.audio_latencies.csv"
            )
            video_latencies.to_csv(
                f"{os.path.splitext(outfile)[0]}.video_latencies.csv"
            )
        avsyncs.to_csv(f"{os.path.splitext(outfile)[0]}.avsync.csv")
        combined.to_csv(f"{os.path.splitext(outfile)[0]}.latencies.csv")
        # combine the lists to one common one
        stats, frame_durations = calculate_stats(
            audio_latencies,
            video_latencies,
            avsyncs,
            video_results,
            errors,
            infile,
            debug,
        )
        if stats is not None:
            stats.to_csv(f"{os.path.splitext(outfile)[0]}.stats.csv")
        else:
            print(f"{infile} failed to produce stats")

        if frame_durations is not None:
            frame_durations.to_csv(f"{infile}.frame_durations.csv")

    return None, None


def calculate_dropped_frames_stats(video, start=-1, stop=-1):
    video = video.dropna()
    if start > 0 or stop > 0:
        video = video.loc[(video["ts"] >= start) & (video["ts"] < stop)]
    if len(video) == 0:
        return 0, 0

    frmin = int(video["video_frame_num_read_int"].min())
    frmax = int(video["video_frame_num_read_int"].max())
    not_in_range = np.setdiff1d(
        range(frmin, frmax), np.unique(video["video_frame_num_read_int"].values)
    )
    frame_count = frmax - frmin
    frames_dropped = len(not_in_range)
    return frame_count, frames_dropped


def dump_frame_drops(video, inputfile):
    # persecond moving average
    start = int(video["ts"].min())
    end = int(video["ts"].max() + 0.5)
    dur = end - start

    framedrops_per_sec = [
        calculate_dropped_frames_stats(video, x, x + 1) for x in range(end - start)
    ]
    fdp = pd.DataFrame(framedrops_per_sec, columns=["frames", "dropped"])
    fdp.to_csv(f"{inputfile}.average.frame.drops.csv")


def calculate_stats(
    audio_latencies,
    video_latencies,
    av_syncs,
    video_results,
    errors,
    inputfile,
    debug=False,
):
    stats = {}
    ignore_latency = False
    if len(av_syncs) == 0 or len(video_results) == 0:
        print(f"Failure - no data")
        return

    stats["file"] = inputfile
    stats["video_latency_sec.mean"] = (
        np.nan
        if len(video_latencies) == 0
        else round(np.mean(video_latencies["video_latency_sec"]), 3)
    )
    stats["video_latency_sec.std_dev"] = (
        np.nan
        if len(video_latencies) == 0
        else round(np.std(video_latencies["video_latency_sec"].values), 3)
    )
    stats["audio_latency_sec.mean"] = (
        np.nan
        if len(video_latencies) == 0
        else round(np.mean(audio_latencies["audio_latency_sec"]), 3)
    )
    stats["audio_latency_sec.std_dev"] = (
        np.nan
        if len(audio_latencies) == 0
        else round(np.std(audio_latencies["audio_latency_sec"].values), 3)
    )
    stats["av_sync_sec.mean"] = round(np.mean(av_syncs["av_sync_sec"]), 3)
    stats["av_sync_sec.std_dev"] = round(np.std(av_syncs["av_sync_sec"].values), 3)

    # Video stats
    # how long time has a frame been shown?
    video = pd.DataFrame(
        video_results,
        columns=[
            "frame_num",
            "ts",
            "expected_frame_num",
            "video_frame_num_read",
            "delta",
        ],
    )
    video["video_frame_num_read_int"] = (
        video["video_frame_num_read"].dropna().astype(int)
    )
    capt_group = video.groupby("video_frame_num_read_int")  # .count()
    cg = capt_group.count()["video_frame_num_read"]
    cg = cg.value_counts().sort_index().to_frame()
    cg.index.rename("consecutive_frames", inplace=True)
    stats["video_frame_times_shows.mean"] = round(capt_group.size().mean(), 2)
    stats["video_frame_times_shows.std_dev"] = round(capt_group.size().std(), 2)
    frame_count, frames_dropped = calculate_dropped_frames_stats(video)
    stats["frames_nbr"] = frame_count
    stats["frames_dropped"] = frames_dropped
    stats["frame_drop.percentage"] = round(100 * frames_dropped / frame_count, 2)
    dump_frame_drops(video, inputfile)
    # errors
    frmin = video["frame_num"].min()
    frmax = video["frame_num"].max()

    failed_frames = 0
    if errors is not None:
        failed_frames = len(errors)
    total_frames = frmax - frmin
    stats["failed_frames"] = failed_frames
    stats["total_frames_parsed"] = total_frames
    stats["frame_parse_error.percentage"] = round(100 * failed_frames / total_frames, 2)
    perrors = pd.DataFrame(errors, columns=["frame", "timestamp", "exception"])
    stats[video_analyze.ERROR_NO_VALID_TAG_MSG] = len(
        perrors.loc[perrors["exception"] == video_analyze.ERROR_NO_VALID_TAG]
    )
    stats[video_analyze.ERROR_INVALID_GRAYCODE_MSG] = len(
        perrors.loc[perrors["exception"] == video_analyze.ERROR_INVALID_GRAYCODE]
    )
    stats[video_analyze.ERROR_SINGLE_GRAYCODE_BIT_MSG] = len(
        perrors.loc[perrors["exception"] == video_analyze.ERROR_SINGLE_GRAYCODE_BIT]
    )
    stats[video_analyze.ERROR_UNKNOWN_MSG] = len(
        perrors.loc[perrors["exception"] == video_analyze.ERROR_UNKNOWN]
    )

    # TODO match gaps with source frame numbers?
    return pd.DataFrame(stats, columns=stats.keys(), index=[0]), cg


def match_video_to_time(
    ts, video_results, beep_period_frames, frame_time, closest=False
):
    # find video frame with ts <= signal ts unless closest
    # then we look both forward and backwards (avsync)
    filt = [x for x in video_results if x[1] <= round(ts, 3)]
    next_val = video_results[len(filt)]
    if len(filt) > 0:

        read_frame_num = filt[-1][3]
        source_frame_num = filt[-1][0]
        if read_frame_num == None or np.isnan(read_frame_num):
            print("read is nan")
            return None
        next_beep_frame = (
            int(read_frame_num / beep_period_frames) + 1
        ) * beep_period_frames
        if closest and next_beep_frame - read_frame_num > beep_period_frames / 2:
            next_beep_frame -= beep_period_frames
        # to next beep minus the time we already watched this frame
        filt = [x for x in filt if x[3] == read_frame_num]
        time_in_frame = ts - filt[0][1]
        latency = (next_beep_frame - read_frame_num) * frame_time - time_in_frame
        if not closest and latency < 0:
            print("ERROR: negative latency")
        else:
            vlat = [
                source_frame_num,
                round(ts, 3),
                read_frame_num,
                next_beep_frame,
                round(latency, 3),
            ]
            return vlat
    return None


def calculate_latency(
    audio_results, video_results, beep_period_sec, audio_offset=0, debug=False
):
    # audio is {sample, ts, cor}
    # video is (frame, ts, expected, read, delta)
    # 1) audio latency is the time between two correlatied values where one should be higher
    # 2) video latency is the time between the frame shown when a signal is played
    # and the time when is should be played out
    # 3) av sync is the difference between when a singal is heard and when the frame is shown

    prev = None
    audio_lat = []
    video_lat = []
    av_sync = []
    combined = []
    beep_period_frames = int(beep_period_sec * 30)  # fps
    frame_time = 1 / 30
    for match in audio_results:
        if prev is not None:
            ts_diff = match[1] - prev[1]
            # correlation indicates that match is an echo (if ts_diff < period)
            if prev[2] > match[2]:
                if ts_diff < beep_period_sec * 0.8:

                    vmatch = match_video_to_time(
                        prev[1],
                        video_results,
                        beep_period_frames,
                        frame_time,
                        closest=False,
                    )
                    if vmatch is not None:
                        vmatch[4] = round(vmatch[4] + audio_offset, 3)
                        video_lat.append(vmatch)
                        audio_lat.append(
                            [
                                prev[0],
                                round(prev[1], 3),
                                vmatch[3],
                                round(ts_diff + audio_offset, 3),
                                prev[2],
                                match[2],
                            ]
                        )
                    avmatch = match_video_to_time(
                        match[1],
                        video_results,
                        beep_period_frames,
                        frame_time,
                        closest=True,
                    )
                    if avmatch is not None:
                        avmatch[4] = round(avmatch[4] + audio_offset, 3)
                        av_sync.append(avmatch)

                    if vmatch is not None and avmatch is not None:
                        combined.append(
                            [
                                vmatch[3],
                                round(ts_diff + audio_offset, 3),
                                vmatch[4],
                                avmatch[4],
                            ]
                        )

        prev = match

    if len(av_sync) == 0:
        # No echo alysis result, this is probably just a av synch measurement
        for match in audio_results:
            vmatch = match_video_to_time(
                match[1], video_results, beep_period_frames, frame_time, closest=True
            )
            if vmatch is not None:
                vmatch[4] = round(vmatch[4] + audio_offset, 3)
                av_sync.append(vmatch)

    audio_lat = pd.DataFrame(
        audio_lat,
        columns=[
            "sample",
            "ts_sec",
            "original_frame",
            "audio_latency_sec",
            "cor1",
            "cor2",
        ],
    )
    video_lat = pd.DataFrame(
        video_lat,
        columns=[
            "frame",
            "ts_sec",
            "frame_num_read",
            "original_frame",
            "video_latency_sec",
        ],
    )
    av_sync = pd.DataFrame(
        av_sync,
        columns=[
            "frame",
            "ts_sec",
            "frame_num_read",
            "original_frame",
            "av_sync_sec",
        ],
    )
    combined = pd.DataFrame(
        combined,
        columns=["frame", "audio_latency_sec", "video_latency_sec", "av_sync_sec"],
    )
    print(f"{audio_lat =}")
    print(f"{video_lat =}")
    print(f"{av_sync=}")
    print(f"{combined =}")
    return audio_lat, video_lat, av_sync, combined


def calculate_latency_old(sync, debug):
    # calculate audio latency, video latency and the ac sync
    # The a/v sync should always be of interest the other two only
    # if there is a transission happening.

    data = []
    sync_points = enumerate(sync)
    item = next(sync_points, None)
    print(f"{sync = }")
    while item != None:
        signal = item[1]
        item = next(sync_points, None)
        if item == None:
            break

        echo = item[1]
        if signal[0] != echo[0]:
            if debug:
                print("Warning: Echo signal is missing")
            signal = echo
            continue

        # First signal time difference will indicate the time delta for video
        # Echo time signal will indicate the a/v sync
        # The time delta between the two will indicate the audio latency
        print(f"{echo[1] = }, {signal[1] =}")
        data.append(
            [
                signal[0],
                int(round(signal[2] * 1000, 0)),
                int(round((echo[1] - signal[1]) * 1000, 0)),
                int(round(echo[2] * 1000, 0)),
            ]
        )
        item = next(sync_points, None)
        signal = None
        echo = None
        exit(0)

    if len(data) == 0:
        # Simple video case
        for signal in sync:
            data.append([signal[0], 0, 0, int(round(signal[2] * 1000, 0))])

    pdata = pd.DataFrame(
        data,
        columns=["video_frame", "video_latency_ms", "audio_latency_ms", "av_sync_ms"],
    )
    return pdata


def dump_results(video_results, video_delta_info, audio_results, outfile, debug):
    # video_results: frame_num, timestamp, frame_num_expected, timestamp, frame_num_read
    # audio_results: sample_num, timestamp, correlation
    # write the output as a csv file
    if video_delta_info is None or len(video_delta_info) == 0:
        print("No video data")
        return
    with open(outfile, "w") as fd:
        fd.write(
            f"timestamp,video_frame_num,video_frame_num_expected,video_frame_num_read,video_delta_frames_{video_delta_info['mode']},audio_sample_num,audio_correlation\n"
        )
        vindex = 0
        aindex = 0
        while vindex < len(video_results) or aindex < len(audio_results):
            # get the timestamps
            vts = video_results[vindex][1] if vindex < len(video_results) else None
            ats = audio_results[aindex][1] if aindex < len(audio_results) else None
            if vts == ats:
                # dump both video and audio entry
                (
                    video_frame_num,
                    timestamp,
                    video_frame_num_expected,
                    video_frame_num_read,
                    video_delta_frames,
                ) = video_results[vindex]
                audio_sample_num, timestamp, audio_correlation = audio_results[aindex]
                vindex += 1
                aindex += 1
            elif ats is None or (vts is not None and vts <= ats):
                # dump a video entry
                (
                    video_frame_num,
                    timestamp,
                    video_frame_num_expected,
                    video_frame_num_read,
                    video_delta_frames,
                ) = video_results[vindex]
                audio_sample_num = audio_correlation = ""
                vindex += 1
            else:
                # dump an audio entry
                audio_sample_num, timestamp, audio_correlation = audio_results[aindex]
                video_frame_num = (
                    video_frame_num_expected
                ) = video_frame_num_read = video_delta_frames = ""
                aindex += 1
            fd.write(
                f"{timestamp},{video_frame_num},{video_frame_num_expected},{video_frame_num_read},{video_delta_frames},{audio_sample_num},{audio_correlation}\n"
            )


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
    # 2-parameter setter using argparse.Action
    parser.add_argument(
        "--width",
        action="store",
        type=int,
        dest="width",
        default=default_values["width"],
        metavar="WIDTH",
        help=("use WIDTH width (default: %i)" % default_values["width"]),
    )
    parser.add_argument(
        "--height",
        action="store",
        type=int,
        dest="height",
        default=default_values["height"],
        metavar="HEIGHT",
        help=("HEIGHT height (default: %i)" % default_values["height"]),
    )

    class VideoSizeAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.width, namespace.height = [int(v) for v in values[0].split("x")]

    parser.add_argument(
        "--video-size",
        action=VideoSizeAction,
        nargs=1,
        help="use <width>x<height>",
    )
    parser.add_argument(
        "--fps",
        action="store",
        type=int,
        dest="fps",
        default=default_values["fps"],
        metavar="FPS",
        help=("use FPS fps (default: %i)" % default_values["fps"]),
    )
    parser.add_argument(
        "--num-frames",
        action="store",
        type=int,
        dest="num_frames",
        default=default_values["num_frames"],
        metavar="NUM_FRAMES",
        help=("use NUM_FRAMES frames (default: %i)" % default_values["num_frames"]),
    )
    parser.add_argument(
        "--pixel-format",
        action="store",
        type=str,
        dest="pixel_format",
        default=default_values["pixel_format"],
        choices=video_common.PIXEL_FORMAT_CHOICES,
        metavar="[%s]"
        % (
            " | ".join(
                video_common.PIXEL_FORMAT_CHOICES,
            )
        ),
        help="pixel format",
    )
    parser.add_argument(
        "--luma-threshold",
        action="store",
        type=int,
        dest="luma_threshold",
        default=default_values["luma_threshold"],
        metavar="LUMA_THRESHOLD",
        help=(
            "detection luma_threshold (default: %i)" % default_values["luma_threshold"]
        ),
    )

    parser.add_argument(
        "--vft-id",
        type=str,
        nargs="?",
        default=default_values["vft_id"],
        choices=vft.VFT_IDS,
        help="%s (default: %s)"
        % (" | ".join("{}".format(k) for k in vft.VFT_IDS), default_values["vft_id"]),
    )
    parser.add_argument(
        "--vft-tag-border-size",
        action="store",
        type=int,
        dest="vft_tag_border_size",
        default=default_values["vft_tag_border_size"],
        metavar="BORDER_SIZE",
        help=(
            "vft tag border size (default: %i)" % default_values["vft_tag_border_size"]
        ),
    )
    parser.add_argument(
        "--pre-samples",
        action="store",
        type=int,
        dest="pre_samples",
        default=default_values["pre_samples"],
        metavar="pre_samples",
        help=("use pre_samples (default: %i)" % default_values["pre_samples"]),
    )
    parser.add_argument(
        "--samplerate",
        action="store",
        type=int,
        dest="samplerate",
        default=default_values["samplerate"],
        metavar="samplerate",
        help=("use samplerate Hz (default: %i)" % default_values["samplerate"]),
    )
    parser.add_argument(
        "--beep-freq",
        action="store",
        type=int,
        dest="beep_freq",
        default=default_values["beep_freq"],
        metavar="beep_freq",
        help=("use beep_freq Hz (default: %i)" % default_values["beep_freq"]),
    )
    parser.add_argument(
        "--beep-duration-samples",
        action="store",
        type=int,
        dest="beep_duration_samples",
        default=default_values["beep_duration_samples"],
        metavar="beep_duration_samples",
        help=(
            "use beep_duration_samples (default: %i)"
            % default_values["beep_duration_samples"]
        ),
    )
    parser.add_argument(
        "--beep-period-sec",
        action="store",
        type=float,
        dest="beep_period_sec",
        default=default_values["beep_period_sec"],
        metavar="beep_period_sec",
        help=("use beep_period_sec (default: %i)" % default_values["beep_period_sec"]),
    )
    parser.add_argument(
        "--scale",
        action="store",
        type=float,
        dest="scale",
        default=default_values["scale"],
        metavar="scale",
        help=("use scale [0-1] (default: %i)" % default_values["scale"]),
    )
    parser.add_argument(
        "--min_separation_msec",
        default=default_values["min_separation_msec"],
        help="Sets a minimal distance between two adjacent signals and sets the shortest detectable time difference in ms. Default is set to half the needle length.",
    )
    parser.add_argument(
        "--correlation_factor",
        default=default_values["correlation_factor"],
        help="Sets the threshold for triggering hits. Default is 10x ratio between the highest correlation and the lower threshold for triggering hits.",
    )
    parser.add_argument(
        "-e",
        "--echo-analysis",
        dest="echo_analysis",
        action="store_true",
        help="Consider multiple hits in order to calculate time between two consecutive audio trigger points. With this a transmission system can be measured for audio and video latency and auiod/video synchronization.",
    )
    parser.add_argument(
        "func",
        type=str,
        nargs="?",
        default=default_values["func"],
        choices=FUNC_CHOICES.keys(),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FUNC_CHOICES.items())),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        dest="infile",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    parser.add_argument(
        "--audio-offset",
        type=float,
        dest="audio_offset",
        default=default_values["audio_offset"],
        metavar="audio_offset",
        help="Adjust problem in sync for either source of measuring device",
    )
    parser.add_argument(
        "--lock-layout",
        action="store_true",
        dest="lock_layout",
        help="With a fixed setup it the first foudn layout can be used for subsequent frames",
    )

    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # implement help
    if options.func == "help":
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)

    # print results
    if options.debug > 2:
        print(options)

    # do something
    if options.func == "generate":
        # get outfile
        if options.outfile == "-":
            options.outfile = "/dev/fd/1"
        assert options.outfile is not None, "error: need a valid output file"
        # do something
        media_file_generate(
            options.width,
            options.height,
            options.fps,
            options.num_frames,
            options.vft_id,
            options.pre_samples,
            options.samplerate,
            options.beep_freq,
            options.beep_duration_samples,
            options.beep_period_sec,
            options.scale,
            options.outfile,
            options.debug,
        )
    elif options.func == "analyze":
        # get infile
        if options.infile == "-":
            options.infile = "/dev/fd/0"
        assert options.infile is not None, "error: need a valid in file"
        # get outfile
        if options.outfile is None or options.outfile == "-":
            options.outfile = "/dev/fd/1"
        video_delta_info, avsync_sec_list = media_file_analyze(
            options.width,
            options.height,
            options.fps,
            options.num_frames,
            options.pixel_format,
            options.luma_threshold,
            options.pre_samples,
            options.samplerate,
            options.beep_freq,
            options.beep_duration_samples,
            options.beep_period_sec,
            options.scale,
            options.infile,
            options.outfile,
            options.debug,
            min_separation_msec=options.min_separation_msec,
            correlation_factor=options.correlation_factor,
            echo_analysis=options.echo_analysis,
            audio_offset=options.audio_offset,
        )
        if not options.echo_analysis:
            if options.debug:
                print(f"{avsync_sec_list = }")
            # print statistics
            if avsync_sec_list:
                avsync_sec_average = np.average(avsync_sec_list)
                avsync_sec_stddev = np.std(avsync_sec_list)
                print(
                    f"avsync_sec average: {avsync_sec_average} stddev: {avsync_sec_stddev} size: {len(avsync_sec_list)}"
                )
            else:
                print("avsync_sec no data available")
            print(f"{video_delta_info = }")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
