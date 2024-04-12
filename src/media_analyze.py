#!/usr/bin/env python

"""media_analyze.py module description."""


import numpy as np
import pandas as pd
import os

import audio_parse
import media_parse
import video_parse
import time


# TODO(chema): implement as a function of media_parse
def media_parse_alt(
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
    lock_layout = kwargs.get("lock_layout", False)
    cache_video = kwargs.get("cache_video", True)
    cache_audio = kwargs.get("cache_audio", True)
    audio_sample = kwargs.get("audio_sample", "")
    tag_manual = kwargs.get("tag_manual", False)
    ref_fps = kwargs.get("ref_fps", -1)
    threaded = kwargs.get("threaded", False)
    # 1. parse the audio stream
    path_audio = f"{infile}.audio.csv"
    if cache_audio and os.path.exists(path_audio):
        audio_results = pd.read_csv(path_audio)
    else:
        # recalculate the audio results
        audio_results = audio_parse.audio_parse(
            infile,
            pre_samples=pre_samples,
            samplerate=samplerate,
            beep_freq=beep_freq,
            beep_duration_samples=beep_duration_samples,
            beep_period_sec=beep_period_sec,
            scale=scale,
            min_separation_msec=kwargs.get("min_separation_msec", 50),
            min_match_threshold=kwargs.get("min_match_threshold", 10),
            audio_sample=audio_sample,
            debug=debug,
        )
        if audio_results is None or len(audio_results) == 0:
            # without audio there is not point in running the video parsing
            raise Exception(
                "ERROR: audio calculation failed. Verify that there are signals in audio stream."
            )
        # write up the results to disk
        audio_results.to_csv(path_audio, index=False)

    # 2. parse the video stream
    path_video = f"{infile}.video.csv"
    if cache_video and os.path.exists(path_video):
        video_results = pd.read_csv(path_video)
    else:
        # recalculate the video results
        video_results = video_parse.video_parse(
            infile,
            width,
            height,
            pixel_format,
            ref_fps,
            luma_threshold,
            lock_layout=lock_layout,
            tag_manual=tag_manual,
            threaded=threaded,
            debug=debug,
        )

        if debug > 0:
            print(
                f"Done parsing, write csv, size: {len(video_results)} to {path_video}"
            )
        # write up the results to disk
        video_results.to_csv(path_video, index=False)
    if debug > 0:
        print("Return video results")
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
    for error, (short, _) in video_parse.ERROR_TYPES.items():
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
    # for error, (short, _) in video_parse.ERROR_TYPES.items():
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
    ts, video_results, beep_period_frames, frame_time, closest=False, debug=0
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
            # look for the previous frame with a valid value_read
            # TODO: maybe interpolate
            # limit the list to half the frame time
            for i in reversed(candidate_list[:-1]):
                if not np.isnan(video_results.iloc[i]["value_read"]):
                    latest_iloc = i
                    latest_frame_num = video_results.iloc[latest_iloc]["frame_num"]
                    latest_value_read = video_results.iloc[latest_iloc]["value_read"]
                    break
                if ts - video_results.iloc[i]["timestamp"] > frame_time / 2:
                    print(f"Could not match {ts} with a frame, too many broken frames")
                    break

            if latest_value_read == None or np.isnan(latest_value_read):
                return None
            if debug > 0:
                print(
                    f"Used previous frame {latest_frame_num} with value {latest_value_read}, {latest_iloc - candidate_list[-1]} frames before"
                )
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
    elif debug > 0:
        print(f"{ts=} not found in video_results")
    return None


def calculate_audio_latency(
    audio_results,
    video_results,
    beep_period_sec,
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
    # Remove echoes.
    audio_latencies["diff"] = audio_latencies["timestamp1"].diff()
    too_close = len(
        audio_latencies.loc[audio_latencies["diff"] < beep_period_sec * 0.5]
    )
    if too_close > 0:
        print(f"WARNING. Potential echoes detected - {too_close} counts")
    audio_latencies.fillna(beep_period_sec, inplace=True)
    audio_latencies = audio_latencies.loc[
        audio_latencies["diff"] > beep_period_sec * 0.5
    ]
    audio_latencies = audio_latencies.drop(columns=["diff"])
    return audio_latencies


def calculate_video_relation(
    audio_latency,
    video_result,
    audio_anchor,
    closest_reference,
    beep_period_sec,
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

        if vmatch is not None and (
            vmatch[4] >= 0 or closest_reference
        ):  # avsync can be negative
            video_latencies.loc[len(video_latencies.index)] = vmatch
        elif vmatch is None:
            print(f"ERROR: no match found for video latency calculation")
        else:
            print(
                f"ERROR: negative video latency - period length needs to be increased, {vmatch}"
            )

    return video_latencies


def calculate_video_latency(
    audio_latency,
    video_result,
    beep_period_sec,
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
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )


def calculate_av_sync(
    audio_data,
    video_result,
    beep_period_sec,
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
        fps=fps,
        ignore_match_order=ignore_match_order,
        debug=debug,
    )
    av_sync = av_sync.rename(columns={"video_latency_sec": "av_sync_sec"})
    return av_sync


def z_filter(data, field, z_val):
    mean = data[field].mean()
    std = data[field].std()
    return data.drop(data[data[field] > mean + z_val * std].index)


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
    calc_coverage,
    vft_id,
    cache_video,
    cache_audio,
    cache_both,
    min_separation_msec,
    min_match_threshold,
    audio_sample,
    lock_layout,
    tag_manual,
    force_fps,
    threaded,
    audio_offset,
    audio_latency,
    video_latency,
    calc_all,
    z_filter,
    av_sync,
    windowed_stats_sec,
    calculate_frame_durations,
    no_hw_decode,
    debug,
):

    if no_hw_decode:
        video_parse.config_decoder(HW_DECODER_ENABLE=False)

    infile_list = []
    # if infile.endswith("]") and infile.startswith("["):
    if infile is not None and "\n" in infile:
        tmp = infile.split("\n")
        infile_list = [infile for infile in tmp if infile != ""]
    else:
        infile_list.append(infile)

    all_audio_latency = pd.DataFrame()
    all_video_latency = pd.DataFrame()
    all_av_sync = pd.DataFrame()
    all_combined = pd.DataFrame()

    quality_stats = pd.DataFrame()
    all_quality_stats = pd.DataFrame()

    all_frame_duration = pd.DataFrame()

    nbr_files = len(infile_list)
    file_cnt = 0
    start = time.monotonic_ns()
    for infile in infile_list:
        # do something
        time_left_sec = np.inf
        estimation = ""
        if file_cnt > 0:
            current_time = time.monotonic_ns()
            time_per_file = (current_time - start) / file_cnt
            print(f"{time_per_file/1000000000.0} sec")
            time_left_sec = time_per_file * (nbr_files - file_cnt) / 1000000000.0
            estimation = f" estimated time left: {time_left_sec:.1f} sec"

        file_cnt += 1
        if infile:
            print(f"---\n({file_cnt}/{nbr_files}) -- {infile} {estimation}")
        outfile = None
        if outfile is not None:
            outfile = outfile.split(".csv")[0]

        if calc_coverage:
            media_parse.media_parse_noise_video(
                infile=infile,
                outfile=outfile,
                vft_id=vft_id,
                debug=debug,
            )
            return

        # get infile
        video_ending = ".video.csv"
        if infile == "-":
            infile = "/dev/fd/0"
        elif infile[-len(video_ending) :] == video_ending:
            # assume the file is using standard naming scheme
            # i.e. *.[mov|mp4].video.csv
            infile = infile[: -len(video_ending)]
        assert infile is not None, "error: need a valid in file"
        # get outfile
        if outfile == "-":
            outfile = "/dev/fd/1"
        cache_video = cache_video and cache_both
        cache_audio = cache_audio and cache_both
        try:
            video_result, audio_result = media_parse_alt(
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
                min_separation_msec=min_separation_msec,
                min_match_threshold=min_match_threshold,
                cache_audio=cache_audio,
                cache_video=cache_video,
                audio_sample=audio_sample,
                lock_layout=lock_layout,
                tag_manual=tag_manual,
                ref_fps=force_fps,  # XXX
                threaded=threaded,
            )
        except Exception as ex:
            print(f"ERROR: {ex} {infile}")
            continue
        audio_latency = None
        video_latency = None
        av_sync = None

        # TODO: capture fps shoud be available
        ref_fps, capture_fps = estimate_fps(video_result)
        if force_fps > 0:
            ref_fps = force_fps

        # Adjust for the audio offset early
        video_result["timestamp"] += audio_offset
        if audio_latency or video_latency or calc_all:
            audio_latency = calculate_audio_latency(
                audio_result,
                video_result,
                fps=ref_fps,
                beep_period_sec=beep_period_sec,
                debug=debug,
            )
        if calc_all or audio_latency:
            if z_filter > 0:
                video_latency = z_filter(audio_latency, "audio_latency_sec", z_filter)
            path = f"{infile}.audio.latency.csv"
            if outfile is not None and len(outfile) > 0 and len(infile_list) == 1:
                path = f"{outfile}.audio.latency.csv"
            audio_latency.to_csv(path, index=False)

        if video_latency or calc_all:
            video_latency = calculate_video_latency(
                audio_latency,
                video_result,
                fps=ref_fps,
                beep_period_sec=beep_period_sec,
                debug=debug,
            )
            if len(video_latency) > 0:
                if z_filter > 0:
                    video_latency = z_filter(
                        video_latency, "video_latency_sec", z_filter
                    )
                path = f"{infile}.video.latency.csv"
                if outfile is not None and len(outfile) > 0 and len(infile_list) == 1:
                    path = f"{outfile}.video.latency.csv"
                video_latency.to_csv(path, index=False)

        if av_sync or calc_all:
            audio_source = audio_result
            if audio_latency is not None and len(audio_latency) > 0:
                audio_source = audio_latency
            av_sync = calculate_av_sync(
                audio_source,
                video_result,
                fps=ref_fps,
                beep_period_sec=beep_period_sec,
                debug=debug,
            )
            if len(av_sync) > 0:
                if z_filter > 0:
                    av_sync = z_filter(av_sync, "av_sync_sec", z_filter)
                path = f"{infile}.avsync.csv"
                if outfile is not None and len(outfile) > 0 and len(infile_list) == 1:
                    path = f"{outfile}.avsync.csv"
                av_sync.to_csv(path, index=False)

        if calc_all:
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
                if outfile is not None and len(outfile) > 0 and len(infile_list) == 1:
                    path = f"{outfile}.latencies.csv"
                combined.to_csv(path, index=False)

        quality_stats = calculate_measurement_quality_stats(audio_result, video_result)
        path = f"{infile}.measurement.quality.csv"
        if outfile is not None and len(outfile) > 0 and len(infile_list) == 1:
            path = f"{outfile}.measurement.quality.csv"
        quality_stats.to_csv(path, index=False)

        if windowed_stats_sec > 0:
            df = calculate_frames_moving_average(video_result, windowed_stats_sec)
            df.to_csv(f"{infile}.video.frames_per_{windowed_stats_sec}sec.csv")

        if calculate_frame_durations:
            df = calculate_frame_durations(video_result)
            df.to_csv(f"{infile}.video.frame_durations.csv")
            if len(infile_list) > 1:
                df["file"] = infile
                all_frame_duration = pd.concat([all_frame_duration, df])

        if len(infile_list) > 1:
            quality_stats["file"] = infile
            all_quality_stats = pd.concat([all_quality_stats, quality_stats])

            # combined data
            if calc_all:
                # only create the combined stat file
                combined["file"] = infile
                all_combined = pd.concat([all_combined, combined])
            else:
                if audio_latency:
                    audio_latency["file"] = infile
                    all_audio_latency = pd.concat([all_audio_latency, audio_latency])
                if video_latency:
                    video_latency["file"] = infile
                    all_video_latency = pd.concat([all_video_latency, video_latency])
                if av_sync:
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
