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
        beep_period_frames,
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
    for audio_ts in audio_results["timestamp"]:
        if video_index >= len(video_results):
            break
        # find the video matches whose timestamps surround the audio one
        while True:
            # find a video entry that has a valid reading
            try:
                prev_video_ts = video_results.iloc[video_index]["timestamp"]
                prev_video_frame_num = video_results.iloc[video_index][
                    "frame_num_expected"
                ]
            except:
                print(f"error: {video_results.iloc[video_index] =}")
                continue
            if prev_video_frame_num is not None:
                break
            video_index += 1
        video_index += 1
        get_next_audio_ts = False
        while not get_next_audio_ts and video_index < len(video_results):
            video_ts = video_results.iloc[video_index]["timestamp"]
            video_frame_num = video_results.iloc[video_index]["frame_num_expected"]
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
            avsync_sec_list.append([video_frame_num, audio_frame_num / fps, avsync_sec])
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
    if debug > 0:
        path_video = f"{infile}.video.csv"
        if os.path.exists(path_video):
            video_results = pd.read_csv(path_video)
        path_video_delta_info = f"{infile}.video.delta_info.csv"
        if os.path.exists(path_video_delta_info):
            video_delta_info = pd.read_csv(path_video_delta_info)

    if video_results is None:
        video_results = video_analyze.video_analyze(
            infile,
            width,
            height,
            fps,
            pixel_format,
            luma_threshold,
            lock_layout=lock_layout,
            debug=debug,
        )
        video_delta_info = video_analyze.video_analyze_delta_info(video_results)
        if video_delta_info is not None and len(video_delta_info) > 0:
            path_video_delta_info = f"{infile}.video.delta_info.csv"
            video_delta_info.to_csv(path_video_delta_info, index=False)
        video_results.to_csv(path_video, index=False)

    # 2. analyze the audio stream
    (
        audio_results,
        audio_duration_samples,
        audio_duration_seconds,
    ) = audio_analyze.audio_analyze(
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
        path_audio = f"{infile}.audio.csv"
        audio_results.to_csv(path_audio, index=False)
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
            path_avsync = f"{infile}.avsync.csv"
            pav_sync_list.to_csv(path_avsync, index=False)

        # 4. dump results to file
        dump_results(video_results, video_delta_info, audio_results, outfile, debug)

        return video_delta_info, avsync_sec_list

    else:
        latencies = []
        # 3b. calculate audio latency, video latency and the difference between the two
        audio_latencies, video_latencies, av_syncs, combined = calculate_latency(
            audio_results, video_results, beep_period_sec, audio_offset, debug
        )
        if debug > 0:
            path_audio_latencies = f"{os.path.splitext(outfile)[0]}.audio.latencies.csv"
            audio_latencies.to_csv(path_audio_latencies, index=False)
            path_video_latencies = f"{os.path.splitext(outfile)[0]}.video.latencies.csv"
            video_latencies.to_csv(path_video_latencies, index=False)
        path_avsync = f"{os.path.splitext(outfile)[0]}.avsync.csv"
        av_syncs.to_csv(path_avsync, index=False)
        path_latencies_combined = f"{os.path.splitext(outfile)[0]}.latencies.csv"
        combined.to_csv(path_latencies_combined, index=False)
        # combine the lists to one common one
        stats, frame_durations = calculate_stats(
            audio_latencies,
            video_latencies,
            av_syncs,
            video_results,
            audio_duration_samples,
            audio_duration_seconds,
            infile,
            debug,
        )
        if stats is not None:
            path_stats = f"{os.path.splitext(outfile)[0]}.stats.csv"
            stats.to_csv(path_stats, index=False)
        else:
            print(f"{infile} failed to produce stats")

        if frame_durations is not None:
            path_frame_durations = f"{infile}.video.frame_durations.csv"
            frame_durations.to_csv(path_frame_durations, index=False)

    return None, None


def calculate_dropped_frames_stats(video_results, start=-1, stop=-1):
    video_results = video_results.dropna()
    if start > 0 or stop > 0:
        video_results = video_results.loc[
            (video_results["timestamp"] >= start) & (video_results["timestamp"] < stop)
        ]
    if len(video_results) == 0:
        return 0, 0

    frmin = int(video_results["value_read_int"].min())
    frmax = int(video_results["value_read_int"].max())
    not_in_range = np.setdiff1d(
        range(frmin, frmax), np.unique(video_results["value_read_int"].values)
    )
    frame_count = frmax - frmin
    frames_unseen = len(not_in_range)
    return frame_count, frames_unseen


def dump_frame_drops(video_results, inputfile):
    # per-second moving average
    start = int(video_results["timestamp"].min())
    end = int(video_results["timestamp"].max() + 0.5)
    dur = end - start
    framedrops_per_sec = [
        (x,) + calculate_dropped_frames_stats(video_results, x, x + 1)
        for x in range(end - start)
    ]
    fdp = pd.DataFrame(
        framedrops_per_sec, columns=["timestamp_start", "frames", "dropped"]
    )
    path_average_frame_drops = f"{inputfile}.video.frame_drops.csv"
    fdp.to_csv(path_average_frame_drops, index=False)


def calculate_stats(
    audio_latencies,
    video_latencies,
    av_syncs,
    video_results,
    audio_duration_samples,
    audio_duration_seconds,
    inputfile,
    debug=False,
):
    stats = {}
    ignore_latency = False
    if len(av_syncs) == 0 or len(video_results) == 0:
        print(f"Failure - no data")
        return None, None

    # 1. basic file statistics
    stats["file"] = inputfile
    video_frames_capture_duration = (
        video_results["timestamp"].max() - video_results["timestamp"].min()
    )
    stats["video_frames_capture_duration_sec"] = video_frames_capture_duration
    video_frames_capture_total = (
        video_results["frame_num"].max() - video_results["frame_num"].min()
    )
    stats["video_frames_capture_total"] = video_frames_capture_total
    stats["audio_frames_capture_duration_frames"] = audio_duration_seconds
    stats["audio_frames_capture_duration_samples"] = audio_duration_samples

    # 2. video latency statistics
    stats["video_latency_sec.num_samples"] = len(video_latencies)
    stats["video_latency_sec.mean"] = (
        np.nan
        if len(video_latencies) == 0
        else np.mean(video_latencies["video_latency_sec"])
    )
    stats["video_latency_sec.std_dev"] = (
        np.nan
        if len(video_latencies) == 0
        else np.std(video_latencies["video_latency_sec"].values)
    )

    # 3. video latency statistics
    stats["audio_latency_sec.num_samples"] = len(audio_latencies)
    stats["audio_latency_sec.mean"] = (
        np.nan
        if len(video_latencies) == 0
        else np.mean(audio_latencies["audio_latency_sec"])
    )
    stats["audio_latency_sec.std_dev"] = (
        np.nan
        if len(audio_latencies) == 0
        else np.std(audio_latencies["audio_latency_sec"].values)
    )

    # 4. avsync statistics
    stats["av_sync_sec.num_samples"] = len(av_syncs)
    stats["av_sync_sec.mean"] = np.mean(av_syncs["av_sync_sec"])
    stats["av_sync_sec.std_dev"] = np.std(av_syncs["av_sync_sec"].values)

    # 5. video source (metiq) stats
    video_results["value_read_int"] = video_results["value_read"].dropna().astype(int)
    dump_frame_drops(video_results, inputfile)
    # 5.1. which source (metiq) frames have been show
    video_frames_sources_min = int(video_results["value_read_int"].min())
    video_frames_sources_max = int(video_results["value_read_int"].max())
    stats["video_frames_source_min"] = video_frames_sources_min
    stats["video_frames_source_max"] = video_frames_sources_max
    (
        video_frames_source_count,
        video_frames_source_unseen,
    ) = calculate_dropped_frames_stats(video_results)
    stats["video_frames_source_total"] = video_frames_source_count
    stats["video_frames_source_seen"] = (
        video_frames_source_count - video_frames_source_unseen
    )
    stats["video_frames_source_unseen"] = video_frames_source_unseen
    stats["video_frames_source_unseen_percentage"] = round(
        100 * video_frames_source_unseen / video_frames_source_count, 2
    )
    # 6. metiq processing statistics
    # TODO(chema): use video.csv information to calculate errors
    # stats["video_frames_metiq_errors"] = video_frames_metiq_errors
    # stats["video_frames_metiq_errors_percentage"] = round(
    #    100 * video_frames_metiq_errors / video_frames_capture_total, 2
    # )
    # video metiq errors
    # for error, (short, _) in video_analyze.ERROR_TYPES.items():
    #     stats["video_frames_metiq_error." + short] = len(
    #         video_metiq_errors.loc[video_metiq_errors["error_type"] == error]
    #     )
    # 7. calculate consecutive frame distribution
    capt_group = video_results.groupby("value_read_int")  # .count()
    cg = capt_group.count()["value_read"]
    cg = cg.value_counts().sort_index().to_frame()
    cg.index.rename("consecutive_frames", inplace=True)
    cg = cg.reset_index()
    # 7.2. times each source (metiq) frame been show
    stats["video_frames_source_appearances.mean"] = capt_group.size().mean()
    stats["video_frames_source_appearances.std_dev"] = capt_group.size().std()

    # TODO match gaps with source frame numbers?
    return pd.DataFrame(stats, columns=stats.keys(), index=[0]), cg


# Function searches for the video_result row whose timestamp
# is closer to ts
# It returns a tuple containing:
# (a) the frame_num of the selected row,
# (b) the searched (input) timestamp,
# (c) the value read in the selected frame,
# (d) the frame_num of the next frame where a beep is expected,
# (e) the latency assuming the initial frame_time.
def match_video_to_time(
    ts, video_results, beep_period_frames, frame_time, closest=False
):
    # get all entries whose ts <= signal ts to a filter
    candidate_list = video_results.index[video_results["timestamp"] <= ts].tolist()
    if len(candidate_list) > 0:
        # check the latest video frame in the filter
        latest_iloc = candidate_list[-1]
        latest_frame_num = video_results.iloc[latest_iloc]["frame_num"]
        latest_value_read = video_results.iloc[latest_iloc]["value_read"]
        if latest_value_read == None or np.isnan(latest_value_read):
            print("read is nan")
            return None
        # estimate the frame for the next beep based on the frequency
        next_beep_frame = (
            int(latest_value_read / beep_period_frames) + 1
        ) * beep_period_frames
        if closest and next_beep_frame - latest_value_read > beep_period_frames / 2:
            next_beep_frame -= beep_period_frames
        # look for other frames where we read the same value
        new_candidate_list = video_results.index[
            video_results["value_read"] == latest_value_read
        ].tolist()
        # get the intersection
        candidate_list = sorted(list(set(candidate_list) & set(new_candidate_list)))
        time_in_frame = ts - video_results.iloc[candidate_list[0]]["timestamp"]
        latency = (next_beep_frame - latest_value_read) * frame_time - time_in_frame
        if not closest and latency < 0:
            print("ERROR: negative latency")
        else:
            vlat = [
                latest_frame_num,
                ts,
                latest_value_read,
                next_beep_frame,
                latency,
            ]
            return vlat
    return None


def calculate_latency(
    audio_results,
    video_results,
    beep_period_sec,
    audio_offset=0,
    ignore_match_order=True,
    debug=False,
):
    # audio is {sample, ts, cor}
    # video is (frame, ts, expected, status, read, delta)
    # 1) audio latency is the time between two correlated values where one should be higher
    # 2) video latency is the time between the frame shown when a signal is played
    # and the time when it should be played out
    # 3) av sync is the difference between when a signal is heard and when the frame is shown

    prev = None
    audio_latencies = []
    video_latencies = []
    av_syncs = []
    combined = []
    # TODO(chema): this assumes the original source was 30 fps
    beep_period_frames = int(beep_period_sec * 30)  # fps
    frame_time = 1 / 30
    # run audio_results looking for audio latency matches,
    # defined as 2x audio correlations that are close and
    # where the correlation value goes down
    for index in range(len(audio_results)):
        if prev is not None:
            match = audio_results.iloc[index]
            ts_diff = match["timestamp"] - prev["timestamp"]
            # correlation indicates that match is an echo (if ts_diff < period)
            if not ignore_match_order and prev["correlation"] <= match["correlation"]:
                continue
            # ensure the 2x correlations are close enough
            if ts_diff >= beep_period_sec * 0.5:
                # default 3 sec -> 1.5 sec, max detected audio delay
                continue
            # audio latency match
            # 1. add audio latency match
            audio_latencies.append(
                [
                    prev["audio_sample"],
                    prev["timestamp"],
                    match["audio_sample"],
                    match["timestamp"],
                    ts_diff,
                    prev["correlation"],
                    match["correlation"],
                ]
            )
            # 2. calculate video latency based on the
            # timestamp of the first (prev) audio match
            # vs. the timestamp of the video frame.
            vmatch = match_video_to_time(
                prev["timestamp"],
                video_results,
                beep_period_frames,
                frame_time,
                closest=False,
            )
            # 3. calculate a/v offset based on the
            # timestamp of the second (match) audio match
            # vs. the timestamp of the video frame.
            avmatch = match_video_to_time(
                match["timestamp"],
                video_results,
                beep_period_frames,
                frame_time,
                closest=True,
            )
            if vmatch is not None:
                # fix the latency using the audio_offset
                vmatch[4] = vmatch[4] + audio_offset
                video_latencies.append(vmatch)
            if avmatch is not None:
                # fix the latency using the audio_offset
                avmatch[4] = avmatch[4] + audio_offset
                av_syncs.append(avmatch)
            if vmatch is not None and avmatch is not None:
                combined.append(
                    [
                        vmatch[3],
                        ts_diff + audio_offset,
                        vmatch[4],
                        avmatch[4],
                    ]
                )
        prev = audio_results.iloc[index]

    if len(av_syncs) == 0:
        # No echo analysis result, this is probably just a av sync measurement
        for index in range(len(audio_results)):
            match = audio_results.iloc[index]
            vmatch = match_video_to_time(
                match["timestamp"],
                video_results,
                beep_period_frames,
                frame_time,
                closest=True,
            )
            if vmatch is not None:
                vmatch[4] = vmatch[4] + audio_offset
                av_syncs.append(vmatch)

    audio_latencies = pd.DataFrame(
        audio_latencies,
        columns=[
            "audio_sample1",
            "timestamp1",
            "audio_sample2",
            "timestamp2",
            "audio_latency_sec",
            "cor1",
            "cor2",
        ],
    )
    video_latencies = pd.DataFrame(
        video_latencies,
        columns=[
            "frame_num",
            "timestamp",
            "frame_num_read",
            "original_frame",
            "video_latency_sec",
        ],
    )
    av_syncs = pd.DataFrame(
        av_syncs,
        columns=[
            "frame_num",
            "timestamp",
            "frame_num_read",
            "original_frame",
            "av_sync_sec",
        ],
    )
    combined = pd.DataFrame(
        combined,
        columns=["frame_num", "audio_latency_sec", "video_latency_sec", "av_sync_sec"],
    )
    if debug > 0:
        print(f"{audio_latencies =}")
        print(f"{video_latencies =}")
        print(f"{av_syncs=}")
        print(f"{combined =}")
    return audio_latencies, video_latencies, av_syncs, combined


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
            f"timestamp,video_frame_num,video_frame_num_expected,video_frame_num_read,video_delta_frames_{video_delta_info.iloc[0]['mode']},audio_sample_num,audio_correlation\n"
        )
        vindex = 0
        aindex = 0
        while vindex < len(video_results) or aindex < len(audio_results):
            # get the timestamps
            video_ts = (
                video_results.iloc[vindex]["timestamp"]
                if vindex < len(video_results)
                else None
            )
            audio_ts = (
                audio_results.iloc[aindex]["timestamp"]
                if aindex < len(audio_results)
                else None
            )
            if video_ts == audio_ts:
                # dump both video and audio entry
                (
                    video_frame_num,
                    video_timestamp,
                    video_frame_num_expected,
                    video_status,
                    video_frame_num_read,
                    video_delta_frames,
                ) = video_results.iloc[vindex]
                (
                    audio_sample_num,
                    audio_timestamp,
                    audio_correlation,
                ) = audio_results.iloc[aindex]
                vindex += 1
                aindex += 1
                timestamp = video_timestamp
            elif audio_ts is None or (video_ts is not None and video_ts <= audio_ts):
                # dump a video entry
                (
                    video_frame_num,
                    video_timestamp,
                    video_frame_num_expected,
                    video_status,
                    video_frame_num_read,
                    video_delta_frames,
                ) = video_results.iloc[vindex]
                audio_sample_num = audio_correlation = ""
                vindex += 1
                timestamp = video_timestamp
            else:
                # dump an audio entry
                (
                    audio_sample_num,
                    audio_timestamp,
                    audio_correlation,
                ) = audio_results.iloc[aindex]
                video_frame_num = (
                    video_frame_num_expected
                ) = video_status = video_frame_num_read = video_delta_frames = ""
                aindex += 1
                timestamp = audio_timestamp
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
        help="Allow multiple audio hits for a single period. This will calculate the time between two consecutive audio trigger points. Feature is useful to include audio and video latency measurements.",
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
        help="Reuse video frame layout location from the first frame to subsequent frames. This reduces the complexity of the analysis when the camera and DUT are set in a fixed setup",
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
