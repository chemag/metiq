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
    "min_match_threshold": audio_analyze.DEFAULT_MIN_MATCH_THRESHOLD,
    # common parameters
    "func": "help",
    "infile": None,
    "outfile": None,
    "audio_offset": 0,
    "lock_layout": False,
}


def media_generate(
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


def media_analyze(
    width,
    height,
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
    audio_offset = kwargs.get("audio_offset", 0)
    lock_layout = kwargs.get("lock_layout", False)
    cache_video = kwargs.get("cache_video", True)
    cache_audio = kwargs.get("cache_audio", True)

    # 1. analyze the audio stream
    path_audio = f"{infile}.audio.csv"
    if cache_audio and os.path.exists(path_audio):
        audio_results = pd.read_csv(path_audio)
    else:
        # recalculate the audio results
        audio_results = audio_analyze.audio_analyze(
            infile,
            pre_samples=pre_samples,
            samplerate=samplerate,
            beep_freq=beep_freq,
            beep_duration_samples=beep_duration_samples,
            beep_period_sec=beep_period_sec,
            scale=scale,
            min_separation_msec=kwargs.get("min_separation_msec", 50),
            min_match_threshold=kwargs.get("min_match_threshold", 10),
            debug=debug,
        )
        if audio_results is None or len(audio_results) == 0:
            # without audio there is not point in running the video analysis
            raise Exception(
                "ERROR: audio calculation failed. Verify that there are signals in audio stream."
            )
        # write up the results to disk
        audio_results.to_csv(path_audio, index=False)

    # 2. analyze the video stream
    path_video = f"{infile}.video.csv"
    if cache_video and os.path.exists(path_video):
        video_results = pd.read_csv(path_video)
    else:
        # recalculate the video results
        video_results = video_analyze.video_parse(
            infile,
            width,
            height,
            pixel_format,
            luma_threshold,
            lock_layout=lock_layout,
            debug=debug,
        )
    # write up the results to disk
    video_results.to_csv(path_video, index=False)

    return video_results, audio_results


def calculate_frames_moving_average(video_result, window_size_sec=1):
    # frame, ts, video_result_frame_num_read_int

    video_result = video_result.dropna()
    if len(video_result) == 0:
        return pd.DataFrame()
    # only one testcase and one situation so no filter is needed.
    startframe = video_result.iloc[0]["value_read"]
    endframe = video_result.iloc[-1]["value_read"]

    frame = startframe
    window_sum = 0
    tmp = 0
    average = []
    while frame < endframe:
        current = video_result.loc[video_result["value_read"] == frame]
        if len(current) == 0:
            frame += 1
            continue
        nextframe = video_result.loc[
            video_result["timestamp"]
            >= (current.iloc[0]["timestamp"] + window_size_sec)
        ]
        if len(nextframe) == 0:
            break

        nextframe_num = nextframe.iloc[0]["value_read"]

        windowed_data = video_result.loc[
            (video_result["value_read"] >= frame)
            & (video_result["value_read"] < nextframe_num)
        ]
        window_sum = len(np.unique(windowed_data["value_read"]))
        distance = nextframe_num - frame
        drops = distance - window_sum
        average.append(
            {
                "frame": frame,
                "frames": distance,
                "shown": window_sum,
                "drops": drops,
                "window": (
                    nextframe.iloc[0]["timestamp"] - current.iloc[0]["timestamp"]
                ),
            }
        )
        frame += 1

    return pd.DataFrame(average)


def estimate_fps(video_result, use_common_fps_vals=True):
    # Estimate source and capture fps by looking at video timestamps
    video_result = video_result.replace([np.inf, -np.inf], np.nan)
    video_result = video_result.dropna(subset=["value_read"])

    if len(video_result) == 0:
        raise Exception("Failed to estimate fps")
    capture_fps = len(video_result) / (
        video_result["timestamp"].max() - video_result["timestamp"].min()
    )

    video_result["value_read_int"] = video_result["value_read"].astype(int)
    min_val = video_result["value_read_int"].min()
    min_ts = video_result.loc[video_result["value_read_int"] == min_val][
        "timestamp"
    ].values[0]
    max_val = video_result["value_read_int"].max()
    max_ts = video_result.loc[video_result["value_read_int"] == max_val][
        "timestamp"
    ].values[0]
    vals = video_result["value_read_int"].unique()

    min_val = np.min(vals)
    max_val = np.max(vals)

    ref_fps = (max_val - min_val) / (max_ts - min_ts)

    common_fps = [7.0, 15.0, 29.97, 30.0, 59.94, 60.0, 119.88, 120.0, 239.76]
    if use_common_fps_vals:
        ref_fps = common_fps[np.argmin([abs(x - ref_fps) for x in common_fps])]
        capture_fps = common_fps[np.argmin([abs(x - capture_fps) for x in common_fps])]
    return ref_fps, capture_fps


def calculate_frame_durations(video_result):
    # Calculate how many times a source frame is shown in capture frames/time
    video_result = video_result.replace([np.inf, -np.inf], np.nan)
    video_result = video_result.dropna(subset=["value_read"])

    ref_fps, capture_fps = estimate_fps(video_result)
    video_result["value_read_int"] = video_result["value_read"].astype(int)
    capt_group = video_result.groupby("value_read_int")
    cg = capt_group.count()["value_read"]
    cg = cg.value_counts().sort_index().to_frame()
    cg.index.rename("consecutive_frames", inplace=True)
    cg["frame_count"] = np.arange(1, len(cg) + 1)
    cg["time"] = cg["frame_count"] / capture_fps
    cg["capture_fps"] = capture_fps
    cg["ref_fps"] = ref_fps
    return cg


def calculate_measurement_quality_stats(audio_result, video_result):
    stats = {}
    frame_errors = video_result.loc[video_result["status"] > 0]
    video_frames_capture_total = len(video_result)

    stats["video_frames_metiq_errors_percentage"] = round(
        100 * len(frame_errors) / video_frames_capture_total, 2
    )

    # video metiq errors
    for error, (short, _) in video_analyze.ERROR_TYPES.items():
        stats["video_frames_metiq_error." + short] = len(
            video_result.loc[video_result["status"] == error]
        )

    # Audio signal
    audio_duration = audio_result["timestamp"].max() - audio_result["timestamp"].min()
    audio_sig_detected = len(audio_result)
    if audio_sig_detected == 0:
        audio_sig_detected = -1  # avoid division by zero
    stats["signal_distance_sec"] = audio_duration / audio_sig_detected
    stats["max_correlation"] = audio_result["correlation"].max()
    stats["min_correlation"] = audio_result["correlation"].min()
    stats["mean_correlation"] = audio_result["correlation"].mean()
    stats["index"] = 0

    return pd.DataFrame(stats, index=[0])


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
            if debug > 0:
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
        if not closest and latency < 0 and debug > 0:
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


def calculate_audio_latency(
    audio_results,
    video_results,
    beep_period_sec,
    audio_offset=0,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # audio is {sample, ts, cor}
    # video is (frame, ts, expected, status, read, delta)
    # audio latency is the time between two correlated values where one should be higher

    prev = None
    audio_latencies = []
    beep_period_frames = int(beep_period_sec * fps)  # fps
    frame_time = 1 / fps
    # run audio_results looking for audio latency matches,
    # defined as 2x audio correlations that are close and
    # where the correlation value goes down
    audio_latencies = pd.DataFrame(
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

    for index in range(len(audio_results)):
        if prev is not None:
            match = audio_results.iloc[index]
            ts_diff = match["timestamp"] - prev["timestamp"]
            # correlation indicates that match is an echo (if ts_diff < period)
            if not ignore_match_order and prev["correlation"] < match["correlation"]:
                # This skip does not move previoua but the next iteration will
                # test agains same prev match
                continue
            # ensure the 2x correlations are close enough
            if ts_diff >= beep_period_sec * 0.5:
                # default 3 sec -> 1.5 sec, max detected audio delay
                prev = match
                continue
            audio_latencies.loc[len(audio_latencies.index)] = [
                prev["audio_sample"],
                prev["timestamp"],
                match["audio_sample"],
                match["timestamp"],
                ts_diff,
                prev["correlation"],
                match["correlation"],
            ]
        prev = audio_results.iloc[index]
    return audio_latencies


def calculate_video_relation(
    audio_latency,
    video_result,
    audio_anchor,
    closest_reference,
    beep_period_sec,
    audio_offset=0,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # video is (frame, ts, expected, status, read, delta)
    # video latency is the time between the frame shown when a signal is played
    # and the time when it should be played out
    prev = None
    video_latencies = []
    beep_period_frames = int(beep_period_sec * fps)  # fps
    frame_time = 1 / fps

    video_latencies = pd.DataFrame(
        columns=[
            "frame_num",
            "timestamp",
            "frame_num_read",
            "original_frame",
            "video_latency_sec",
        ],
    )

    for index in range(len(audio_latency)):
        match = audio_latency.iloc[index]
        # calculate video latency based on the
        # timestamp of the first (prev) audio match
        # vs. the timestamp of the video frame.
        vmatch = match_video_to_time(
            match[audio_anchor],
            video_result,
            beep_period_frames,
            frame_time,
            closest=closest_reference,
        )
        if vmatch is not None:
            # fix the latency using the audio_offset
            vmatch[4] = vmatch[4] + audio_offset
            video_latencies.loc[len(video_latencies.index)] = vmatch

    return video_latencies


def calculate_video_latency(
    audio_latency,
    video_result,
    beep_period_sec,
    audio_offset=0,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # video latency is the time between the frame shown when a signal is played
    return calculate_video_relation(
        audio_latency,
        video_result,
        "timestamp1",
        False,
        beep_period_sec=beep_period_sec,
        audio_offset=audio_offset,
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )


def calculate_av_sync(
    audio_data,
    video_result,
    beep_period_sec,
    audio_offset=0,
    fps=30,
    ignore_match_order=True,
    debug=False,
):
    # av sync is the difference between when a signal is heard and when the frame is shown
    # If there is a second ssignal, use that one.
    timefield = "timestamp2"
    if timefield not in audio_data.columns:
        timefield = "timestamp"
    av_sync = calculate_video_relation(
        audio_data,
        video_result,
        timefield,
        True,
        beep_period_sec=beep_period_sec,
        audio_offset=audio_offset,
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )
    av_sync = av_sync.rename(columns={"video_latency_sec": "av_sync_sec"})
    return av_sync


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
        "--force-fps",
        action="store",
        type=int,
        dest="force_fps",
        default=-1,
        metavar="ForceFPS",
        help=(
            "If the auto detection mechanism failes, this value can be used to override te measured value."
        ),
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
        "--min-separation-msec",
        dest="min_separation_msec",
        default=default_values["min_separation_msec"],
        help="Sets a minimal distance between two adjacent signals and sets the shortest detectable time difference in ms. Default is set to half the needle length.",
    )
    parser.add_argument(
        "--min-match-threshold",
        dest="min_match_threshold",
        default=default_values["min_match_threshold"],
        help=f"Sets the threshold for detecting audio matches. Default is {default_values['min_match_threshold']}, ratio between the highest correlation and the lower threshold for triggering hits.",
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
        required=False,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache_both",
        default=True,
        help="Recalculate both audio and video analysis",
    )
    parser.add_argument(
        "--no-cache-audio",
        action="store_false",
        dest="cache_audio",
        default=True,
        help="Recalculate audio analysis",
    )
    parser.add_argument(
        "--no-cache-video",
        action="store_false",
        dest="cache_video",
        default=True,
        help="Recalculate video analysis",
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
    parser.add_argument(
        "--calc-all",
        action="store_true",
        dest="calc_all",
        help="Calculate all possible derived data",
    )
    parser.add_argument(
        "--audio-latency",
        action="store_true",
        dest="audio_latency",
        help="Calculate audio latency.",
    )
    parser.add_argument(
        "--video-latency",
        action="store_true",
        dest="video_latency",
        help="Calculate video latency.",
    )
    parser.add_argument(
        "--av-sync",
        action="store_true",
        dest="av_sync",
        help="Calculate audio/video synchronization using audio timestamps and video frame numbers.",
    )
    parser.add_argument(
        "--windowed-stats-sec",
        type=float,
        dest="windowed_stats_sec",
        default=-1,
        help="Calculate video frames shown/dropped per unit sec.",
    )
    parser.add_argument(
        "--calc-frame-durations",
        action="store_true",
        dest="calculate_frame_durations",
        help="Calculate source frame durations.",
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

    files = []
    # if options.infile.endswith("]") and options.infile.startswith("["):
    if "\n" in options.infile:
        tmp = options.infile.split("\n")
        files = [fil for fil in tmp if fil != ""]
    else:
        files.append(options.infile)

    all_audio_latency = pd.DataFrame()
    all_video_latency = pd.DataFrame()
    all_av_sync = pd.DataFrame()
    all_combined = pd.DataFrame()

    quality_stats = pd.DataFrame()
    all_quality_stats = pd.DataFrame()

    all_frame_duration = pd.DataFrame()

    for infile in files:
        # do something
        outfile = None
        if options.outfile is not None:
            outfile = options.outfile.split(".csv")[0]
        if options.func == "generate":
            # get outfile
            if options.outfile == "-":
                outfile = "/dev/fd/1"
            # do something
            # do something
            media_generate(
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
                outfile,
                options.debug,
            )
        elif options.func == "analyze":
            # get infile
            if infile == "-":
                infile = "/dev/fd/0"
            assert infile is not None, "error: need a valid in file"
            # get outfile
            if options.outfile == "-":
                outfile = "/dev/fd/1"
            cache_video = options.cache_video and options.cache_both
            cache_audio = options.cache_audio and options.cache_both
            try:
                video_result, audio_result = media_analyze(
                    options.width,
                    options.height,
                    options.num_frames,
                    options.pixel_format,
                    options.luma_threshold,
                    options.pre_samples,
                    options.samplerate,
                    options.beep_freq,
                    options.beep_duration_samples,
                    options.beep_period_sec,
                    options.scale,
                    infile,
                    outfile,
                    options.debug,
                    min_separation_msec=options.min_separation_msec,
                    min_match_threshold=options.min_match_threshold,
                    audio_offset=options.audio_offset,
                    cache_audio=cache_audio,
                    cache_video=cache_video,
                )
            except Exception as ex:
                print(f"ERROR: {ex} {infile}")
                continue
            audio_latency = None
            video_latency = None
            av_sync = None

            ref_fps, capture_fps = estimate_fps(video_result)
            if options.force_fps > 0:
                ref_fps = options.force_fps

            if options.audio_latency or options.video_latency or options.calc_all:
                audio_latency = calculate_audio_latency(
                    audio_result,
                    video_result,
                    fps=ref_fps,
                    beep_period_sec=options.beep_period_sec,
                    audio_offset=options.audio_offset,
                    debug=options.debug,
                )
            if options.calc_all or options.audio_latency:
                path = f"{infile}.audio.latency.csv"
                if outfile is not None and len(outfile) > 0 and len(files) == 1:
                    path = f"{outfile}.audio.latency.csv"
                audio_latency.to_csv(path, index=False)

            if options.video_latency or options.calc_all:
                video_latency = calculate_video_latency(
                    audio_latency,
                    video_result,
                    fps=ref_fps,
                    beep_period_sec=options.beep_period_sec,
                    audio_offset=options.audio_offset,
                    debug=options.debug,
                )
                if len(video_latency) > 0:
                    path = f"{infile}.video.latency.csv"
                    if outfile is not None and len(outfile) > 0 and len(files) == 1:
                        path = f"{outfile}.video.latency.csv"
                    video_latency.to_csv(path, index=False)

            if options.av_sync or options.calc_all:
                audio_source = audio_result
                if audio_latency is not None and len(audio_latency) > 0:
                    audio_source = audio_latency
                av_sync = calculate_av_sync(
                    audio_source,
                    video_result,
                    fps=ref_fps,
                    beep_period_sec=options.beep_period_sec,
                    debug=options.debug,
                )
                if len(av_sync) > 0:
                    path = f"{infile}.avsync.csv"
                    if outfile is not None and len(outfile) > 0 and len(files) == 1:
                        path = f"{outfile}.avsync.csv"
                    av_sync.to_csv(path, index=False)

            if options.calc_all:
                # video latency and avsync latency share original frame
                # video latency and audio latency share timestamp

                combined = []
                frames = video_latency["original_frame"].values
                for frame in frames:
                    video_latency_row = video_latency.loc[
                        video_latency["original_frame"] == frame
                    ]
                    video_latency_sec = video_latency_row["video_latency_sec"].values[0]
                    timestamp = video_latency_row["timestamp"].values[0]
                    audio_latency_row = audio_latency.loc[
                        audio_latency["timestamp1"] == timestamp
                    ]

                    if len(audio_latency_row) == 0:
                        continue
                    audio_latency_sec = audio_latency_row["audio_latency_sec"].values[0]
                    av_sync_sec_row = av_sync.loc[av_sync["original_frame"] == frame]
                    if len(av_sync_sec_row) == 0:
                        continue

                    av_sync_sec = av_sync_sec_row["av_sync_sec"].values[0]
                    combined.append(
                        [frame, audio_latency_sec, video_latency_sec, av_sync_sec]
                    )

                combined = pd.DataFrame(
                    combined,
                    columns=[
                        "frame_num",
                        "audio_latency_sec",
                        "video_latency_sec",
                        "av_sync_sec",
                    ],
                )

                if len(combined) > 0:
                    path = f"{infile}.latencies.csv"
                    if outfile is not None and len(outfile) > 0 and len(files) == 1:
                        path = f"{outfile}.latencies.csv"
                    combined.to_csv(path, index=False)

            quality_stats = calculate_measurement_quality_stats(
                audio_result, video_result
            )
            path = f"{infile}.measurement.quality.csv"
            if outfile is not None and len(outfile) > 0 and len(files) == 1:
                path = f"{outfile}.measurement.quality.csv"
            quality_stats.to_csv(path, index=False)

            if options.windowed_stats_sec > 0:
                df = calculate_frames_moving_average(
                    video_result, options.windowed_stats_sec
                )
                df.to_csv(
                    f"{infile}.video.frames_per_{options.windowed_stats_sec}sec.csv"
                )

            if options.calculate_frame_durations:
                df = calculate_frame_durations(video_result)
                df.to_csv(f"{infile}.video.frame_durations.csv")
                if len(files) > 1:
                    df["file"] = infile
                    all_frame_duration = pd.concat([all_frame_duration, df])

            if len(files) > 1:
                quality_stats["file"] = infile
                all_quality_stats = pd.concat([all_quality_stats, quality_stats])

                # combined data
                if options.calc_all:
                    # only create the combined stat file
                    combined["file"] = infile
                    all_combined = pd.concat([all_combined, combined])
                else:
                    if options.audio_latency:
                        audio_latency["file"] = infile
                        all_audio_latency = pd.concat(
                            [all_audio_latency, audio_latency]
                        )
                    if options.video_latency:
                        video_latency["file"] = infile
                        all_video_latency = pd.concat(
                            [all_video_latency, video_latency]
                        )
                    if options.av_sync:
                        av_sync["file"] = infile
                        all_av_sync = pd.concat([all_av_sync, av_sync])

            # print statistics
            if av_sync is not None:
                avsync_sec_average = np.average(av_sync["av_sync_sec"])
                avsync_sec_stddev = np.std(av_sync["av_sync_sec"])
                print(
                    f"avsync_sec average: {avsync_sec_average} stddev: {avsync_sec_stddev} size: {len(av_sync)}"
                )

    if len(all_audio_latency) > 0:
        path = f"all.audio_latency.csv"
        if outfile is not None and len(outfile) > 0:
            path = f"{outfile}.all.audio_latency.csv"
        all_audio_latency.to_csv(path, index=False)

    if len(all_video_latency) > 0:
        path = f"all.video_latency.csv"
        if outfile is not None and len(outfile) > 0:
            path = f"{outfile}.all.video_latency.csv"
        all_video_latency.to_csv(path, index=False)

    if len(all_av_sync) > 0:
        path = f"all.avsync.csv"
        if outfile is not None and len(outfile) > 0:
            path = f"{outfile}.all.avsync.csv"
        all_av_sync.to_csv(path, index=False)

    if len(all_combined) > 0:
        path = f"all.combined.csv"
        if outfile is not None and len(outfile) > 0:
            path = f"{outfile}.all.latencies.csv"
        all_combined.to_csv(path, index=False)

    if len(all_quality_stats) > 0:
        path = f"all.quality_stats.csv"
        if outfile is not None and len(outfile) > 0:
            path = f"{outfile}.all.measurement.quality.csv"
        all_quality_stats.to_csv(path, index=False)

    if len(all_frame_duration) > 0:
        path = f"all.frame_duration.csv"
        if outfile is not None and len(outfile) > 0:
            path = f"{outfile}.all.frame_duration.csv"
        all_frame_duration.to_csv(path, index=False)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
