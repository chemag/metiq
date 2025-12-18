#!/usr/bin/env python3

"""metiq_reader_mp4box.py: MP4Box-based container analysis.

This module provides MP4ContainerAnalyzer that uses MP4Box -diso to analyze
the MP4/ISOBMFF container structure, extracting timing information from
edts/elst (Edit List) boxes.

The Edit List box (elst) can contain entries that:
1. Add silence at the beginning (media_time = -1)
2. Skip samples at the beginning (media_time > 0)
3. Normal playback (media_time = 0)

Example edts with 2402-sample silence:
    <EditBox Size="48" Type="edts">
      <EditListBox Size="40" Type="elst" EntryCount="2">
        <EditListEntry Duration="2402" MediaTime="-1" MediaRate="1"/>
        <EditListEntry Duration="2999179" MediaTime="0" MediaRate="1"/>
      </EditListBox>
    </EditBox>

Example edts with 2112-sample skip:
    <EditBox Size="28" Type="edts">
      <EditListBox Size="16" Type="elst" EntryCount="1">
        <EditListEntry Duration="14404810" MediaTime="2112" MediaRate="1"/>
      </EditListBox>
    </EditBox>
"""

import argparse
import subprocess
import sys
import tempfile
import typing
import xml.etree.ElementTree as ET


default_values = {
    "debug": 0,
}


class EditListEntry:
    """Represents a single entry in an Edit List (elst) box."""

    def __init__(
        self,
        segment_duration: int,
        media_time: int,
        media_rate: float,
    ):
        """Initialize an Edit List entry.

        Args:
            segment_duration: Duration of this edit segment in movie timescale units.
            media_time: Starting time within the media (-1 for empty edit/silence,
                        0 for normal playback from start, >0 to skip samples).
            media_rate: Playback rate (typically 1.0).
        """
        self.segment_duration = segment_duration
        self.media_time = media_time
        self.media_rate = media_rate

    def __repr__(self) -> str:
        return (
            f"EditListEntry(duration={self.segment_duration}, "
            f"media_time={self.media_time}, rate={self.media_rate})"
        )


class TrackTimingInfo:
    """Container timing information for a single track."""

    def __init__(self, track_id: int, timescale: int, movie_timescale: int):
        """Initialize track timing info.

        Args:
            track_id: Track identifier.
            timescale: Track timescale (samples per second for audio, units for video).
            movie_timescale: Movie timescale (used for edit list durations/times).
        """
        self.track_id = track_id
        self.timescale = timescale
        self.movie_timescale = movie_timescale
        self.edit_list: typing.List[EditListEntry] = []

    def get_start_time_samples(self) -> int:
        """Get start time offset in movie timescale units (from silence/empty edits).

        Returns:
            Number of movie timescale units of silence prepended (0 if none).
            This corresponds to edts entries with media_time=-1.
        """
        start_samples = 0
        for entry in self.edit_list:
            if entry.media_time == -1:
                # Empty edit (silence) - add its duration (in movie timescale units)
                start_samples += entry.segment_duration
        return start_samples

    def get_skip_time_samples(self) -> int:
        """Get skip time offset in movie timescale units (from media_time > 0).

        Returns:
            Number of movie timescale units to skip at the beginning (0 if none).
            This corresponds to edts entries with media_time > 0.
        """
        skip_samples = 0
        for entry in self.edit_list:
            if entry.media_time > 0:
                # Skip edit - use the media_time value
                skip_samples = entry.media_time
                break  # Use first skip entry
        return skip_samples

    def get_start_time_track_samples(self) -> int:
        """Get start time offset in track timescale samples.

        Returns:
            Number of track samples of silence prepended (0 if none).
            Converts from movie timescale to track timescale.
        """
        if self.movie_timescale == 0:
            return 0
        movie_samples = self.get_start_time_samples()
        # Convert: movie_units / movie_timescale * track_timescale
        return int(movie_samples * self.timescale / self.movie_timescale)

    def get_start_time_seconds(self) -> float:
        """Get start time offset in seconds.

        Returns:
            Start time in seconds (0.0 if no offset).
        """
        if self.movie_timescale == 0:
            return 0.0
        return self.get_start_time_samples() / self.movie_timescale

    def get_skip_time_seconds(self) -> float:
        """Get skip time offset in seconds.

        Returns:
            Skip time in seconds (0.0 if no offset).
        """
        if self.movie_timescale == 0:
            return 0.0
        return self.get_skip_time_samples() / self.movie_timescale

    def __repr__(self) -> str:
        return (
            f"TrackTimingInfo(track_id={self.track_id}, "
            f"timescale={self.timescale}, "
            f"movie_timescale={self.movie_timescale}, "
            f"entries={len(self.edit_list)}, "
            f"start_time_samples={self.get_start_time_samples()}, "
            f"skip_time_samples={self.get_skip_time_samples()}, "
            f"start_time_sec={self.get_start_time_seconds():.3f}, "
            f"skip_time_sec={self.get_skip_time_seconds():.3f})"
        )


class MP4ContainerAnalyzer:
    """Analyzes MP4/ISOBMFF container structure using MP4Box.

    Uses MP4Box -diso to generate XML output, then parses it to extract
    timing information from edts/elst boxes.
    """

    def __init__(self, input_file: str, debug: int = 0):
        """Initialize the MP4 container analyzer.

        Args:
            input_file: Path to the MP4/ISOBMFF file.
            debug: Debug level (0=quiet, higher=more verbose).
        """
        self.input_file = input_file
        self.debug = debug
        self._tracks: typing.Dict[int, TrackTimingInfo] = {}
        self._movie_timescale: int = 0
        self._analyzed = False

    def analyze(self) -> bool:
        """Analyze the MP4 container structure.

        Returns:
            True if analysis succeeded, False otherwise.
        """
        if self._analyzed:
            return len(self._tracks) > 0

        self._analyzed = True

        # Create temporary file for XML output
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tmp:
            xml_file = tmp.name

        try:
            # Run MP4Box -diso to generate XML output
            cmd = ["MP4Box", "-diso", self.input_file, "-out", xml_file]

            if self.debug > 0:
                print(f"MP4ContainerAnalyzer: running {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                if self.debug > 0:
                    print(f"MP4ContainerAnalyzer: MP4Box failed: {result.stderr}")
                return False

            # Parse XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find MovieHeaderBox to get movie timescale
            movie_header = root.find(".//{*}MovieHeaderBox")
            if movie_header is not None:
                self._movie_timescale = int(movie_header.get("TimeScale", "0"))
                if self.debug > 0:
                    print(
                        f"MP4ContainerAnalyzer: movie timescale = {self._movie_timescale}"
                    )
            else:
                if self.debug > 0:
                    print("MP4ContainerAnalyzer: warning: no MovieHeaderBox found")
                return False

            # Find all TrackBox elements
            # The structure is: MovieBox > TrackBox > [TrackHeaderBox, EditBox, MediaBox]
            for track_box in root.findall(".//{*}TrackBox"):
                self._parse_track(track_box)

            if self.debug > 0:
                print(
                    f"MP4ContainerAnalyzer: found {len(self._tracks)} tracks "
                    f"with timing info"
                )
                for track_info in self._tracks.values():
                    print(f"  {track_info}")

            return len(self._tracks) > 0

        except Exception as e:
            if self.debug > 0:
                print(f"MP4ContainerAnalyzer: analysis failed: {e}")
            return False

        finally:
            # Clean up temporary file
            import os

            try:
                os.unlink(xml_file)
            except Exception:
                pass

    def _parse_track(self, track_box: ET.Element) -> None:
        """Parse a single TrackBox element.

        Args:
            track_box: TrackBox XML element.
        """
        # Find TrackHeaderBox to get track ID
        track_header = track_box.find("{*}TrackHeaderBox")
        if track_header is None:
            return

        track_id = int(track_header.get("TrackID", "0"))
        if track_id == 0:
            return

        # Find MediaBox > MediaHeaderBox to get timescale
        media_box = track_box.find("{*}MediaBox")
        if media_box is None:
            return

        media_header = media_box.find("{*}MediaHeaderBox")
        if media_header is None:
            return

        timescale = int(media_header.get("TimeScale", "0"))
        if timescale == 0:
            return

        # Create track timing info
        track_info = TrackTimingInfo(track_id, timescale, self._movie_timescale)

        # Find EditBox > EditListBox if present
        edit_box = track_box.find("{*}EditBox")
        if edit_box is not None:
            edit_list_box = edit_box.find("{*}EditListBox")
            if edit_list_box is not None:
                # Parse all EditListEntry elements
                for entry_elem in edit_list_box.findall("{*}EditListEntry"):
                    duration = int(entry_elem.get("Duration", "0"))
                    media_time = int(entry_elem.get("MediaTime", "0"))
                    media_rate = float(entry_elem.get("MediaRate", "1.0"))

                    entry = EditListEntry(duration, media_time, media_rate)
                    track_info.edit_list.append(entry)

        # Store track info
        self._tracks[track_id] = track_info

    def get_audio_timing_info(self) -> typing.Optional[TrackTimingInfo]:
        """Get timing information for the first audio track.

        Returns:
            TrackTimingInfo for audio track, or None if no audio track found.
        """
        if not self._analyzed:
            self.analyze()

        # For now, return the first track we find (typically track 1 or 2)
        # In the future, we could identify audio tracks by type
        # For most cases, audio is track 2, but we'll return the first with timing info
        if len(self._tracks) == 0:
            return None

        # Return first track (usually this works for simple cases)
        # TODO: Could enhance this to specifically identify audio tracks
        track_ids = sorted(self._tracks.keys())
        for track_id in track_ids:
            # Skip track 1 if track 2 exists (track 2 is often audio)
            if len(track_ids) > 1 and track_id == 1:
                continue
            return self._tracks[track_id]

        # Fallback to first track
        return self._tracks[track_ids[0]]

    def get_video_timing_info(self) -> typing.Optional[TrackTimingInfo]:
        """Get timing information for the first video track.

        Returns:
            TrackTimingInfo for video track, or None if no video track found.
        """
        if not self._analyzed:
            self.analyze()

        # For now, return track 1 if it exists (typically video)
        # TODO: Could enhance this to specifically identify video tracks
        if 1 in self._tracks:
            return self._tracks[1]

        # Fallback to first track
        if len(self._tracks) > 0:
            track_ids = sorted(self._tracks.keys())
            return self._tracks[track_ids[0]]

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
    parser.add_argument("input_file", help="Path to MP4 file to analyze")
    # do the parsing
    options = parser.parse_args(argv[1:])
    return options


def main():
    """Simple test of MP4ContainerAnalyzer."""
    options = get_options(sys.argv)

    # Normalize debug level (use 0 if negative)
    debug_level = max(0, options.debug)

    analyzer = MP4ContainerAnalyzer(options.input_file, debug=debug_level)
    if analyzer.analyze():
        audio_info = analyzer.get_audio_timing_info()
        if audio_info:
            print(f"audio: {audio_info}")
            print(f"movie_timescale: {audio_info.movie_timescale}")
            print(f"audio_timescale: {audio_info.timescale}")
            print(f"audio_start_samples: {audio_info.get_start_time_samples()}")
            print(f"audio_skip_samples: {audio_info.get_skip_time_samples()}")
            print(f"audio_start_sec: {audio_info.get_start_time_seconds():.6f}")
            print(f"audio_skip_sec: {audio_info.get_skip_time_seconds():.6f}")
        else:
            print("  No audio track found")

        video_info = analyzer.get_video_timing_info()
        if video_info:
            print(f"video: {video_info}")
            print(f"movie_timescale: {audio_info.movie_timescale}")
            print(f"video_timescale: {video_info.timescale}")
            print(f"video_start_samples: {video_info.get_start_time_samples()}")
            print(f"video_skip_samples: {video_info.get_skip_time_samples()}")
            print(f"video_start_sec: {video_info.get_start_time_seconds():.6f}")
            print(f"video_skip_sec: {video_info.get_skip_time_seconds():.6f}")
        else:
            print("  No video track found")
    else:
        print("Failed to analyze container")
        sys.exit(1)


if __name__ == "__main__":
    main()
