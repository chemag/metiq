# METIQ - Media Timing Quality Measurement Tool

Metiq is a tool to measure timing distortions caused by media paths. It measures
(a) A/V sync (audio-to-video synchronization), (b) video smoothness (frame drops,
duplications, jankiness), and can be adapted to measure end-to-end latency.

Version: 0.4


# 1. Project Purpose

Metiq provides a way to inject a reference media file into a system, capture the
output as a "distorted" file, and then measure timing distortions between reference
and distorted media. Unlike video quality tools (VMAF) or audio quality tools
(visqol), metiq focuses solely on timing accuracy.


# 2. Core Concepts

# 2.1. A/V Sync

Audio-to-video synchronization (lipsync) measures the offset between audio and
video streams. A positive value means audio is earlier than video; negative means
audio is later. Industry standards define acceptable ranges:
- Steinmetz: +/-80 ms acceptable
- EBU R37-2007: +40 ms to -60 ms good range
- ITU-R BT.1359-1: +45 ms to -125 ms detectable, +90 ms to -185 ms acceptable

# 2.2. Video Smoothness

Measures whether video frames are rendered in correct order. Issues arise from:
- Framerate conversions (lazy or interpolation-based)
- VSYNC adaptation between producers/consumers
- Camera/display frequency mismatches


# 3. Technical Innovation: VFT Codes

VFT (Video Fine-Grained Time-Mix-Resistant 2D Barcodes) are custom barcodes
designed to survive frame interpolation. Key features:
- Uses Gray codes where only one bit changes between consecutive values
- Handles interpolated frames by detecting "undecided" bits
- Contains ArUco fiducial markers for positional robustness
- Multiple formats: 5x4 (8 bits), 7x5 (16 bits, default), 9x6 (25 bits), 9x8 (34 bits)


# 4. Project Structure

# 4.1. Main CLI Entry Point

- src/metiq.py (678 lines) - Main CLI with three subcommands: generate, parse, analyze

# 4.2. Core Modules

- src/media_generate.py (111 lines) - Orchestrates reference media generation
- src/media_parse.py (186 lines) - Orchestrates distorted media analysis
- src/media_analyze.py (1027 lines) - Statistical analysis of parsed results

# 4.3. Video Processing

- src/vft.py (907 lines) - VFT barcode generation and parsing with Gray codes
- src/video_generate.py (323 lines) - Video frame generation with VFT codes
- src/video_parse.py (1017 lines) - Video frame extraction and analysis
- src/video_common.py (60 lines) - Video constants and ImageInfo dataclass
- src/aruco_common.py (83 lines) - ArUco marker generation and detection
- src/video_tag_coordinates.py (140 lines) - Interactive tag coordinate selection UI

# 4.4. Audio Processing

- src/audio_generate.py - Audio beep pattern generation
- src/audio_parse.py (398 lines) - Audio signal correlation and beep detection
- src/audio_common.py (165 lines) - Audio generation functions (sin, chirp, beep)

# 4.5. Utilities

- src/common.py (45 lines) - Shell command execution wrapper
- src/media_plot.py (431 lines) - Visualization and plotting of results
- src/metiq_multi.py (702 lines) - Multi-file parallel processing
- src/_version.py (7 lines) - Version string

# 4.6. Tests

- test/tests.py - E2E test suite
- test/verify.py - Verification utilities
- test/verify_generate.py - Test file generation
- test/verify_config.py - Test configuration
- test/verify_unittests.py - Unit tests


# 5. Workflow

# 5.1. Generate Reference

Create a reference video with embedded timing markers:
```
./metiq.py generate -o reference.mp4
```

Video options: --width, --height, --fps, --num-frames, --vft-id
Audio options: --samplerate, --beep-freq, --beep-duration-samples, --beep-period-sec

# 5.2. Run Experiment

Play reference through device under test (DUT) and capture output. Examples:
- Display/camera testing: play on display, capture with camera
- VC system testing: inject into VC, capture rendered output
- Bluetooth speaker testing: compare internal vs external speaker

# 5.3. Parse Distorted File

Analyze the captured output:
```
./metiq.py parse -i distorted.mp4 -o output.csv
```

Key options:
- --luma-threshold: bit recognition threshold (20 for camera, 100 for processing)
- --lock-layout: fix VFT position
- --threaded: parallel frame decoding

# 5.4. Analyze Results

Generate statistics from parsed data:
```
./metiq.py analyze --input-audio audio.csv --input-video video.csv -a all
```

Options: --audio-offset, --z-filter, --windowed-stats-sec


# 6. Output Metrics

# 6.1. A/V Sync Results

- average: mean offset in seconds
- stddev: standard deviation
- size: number of valid samples

# 6.2. Video Smoothness Results

- mode: baseline frame number difference
- stddev: jitter measurement
- ok_ratio: frames matching expected number (95%+ is good)
- sok_ratio: frames within 0.5 frame distance
- nok_ratio: significantly mismatched frames
- unknown_ratio: unreadable frames


# 7. Dependencies

From requirements.txt:
- numpy - Array operations
- scipy - Signal processing (FFT, filtering)
- graycode - Gray code encoding/decoding
- opencv-python - Video capture and image processing
- opencv-contrib-python - ArUco marker detection
- pandas - Data manipulation and CSV output
- Shapely - Geometric calculations
- matplotlib - Plotting and visualization

External: ffmpeg (for video encoding/decoding)


# 8. Key Design Decisions

# 8.1. Gray Code for Frame Mixing Resilience

Gray codes guarantee only one bit changes between consecutive values, allowing
detection of interpolated frames without complete data loss.

# 8.2. Fiducial Markers for Camera Positioning

ArUco markers in corners enable affine transformation recovery when camera
positioning is imperfect.

# 8.3. Audio Beep Correlation

Uses signal correlation to robustly detect periodic audio beeps even after
processing distortions.


# 9. Common Commands

Generate 30fps reference video:
```
./metiq.py generate -o ref.mp4 --fps 30 --num-frames 600
```

Parse with camera capture settings:
```
./metiq.py parse -i captured.mp4 -o results.csv --luma-threshold 20
```

Run multi-file processing:
```
python src/metiq_multi.py --input-dir captures/ --output-dir results/
```


# 10. File Format Notes

Output CSV files contain:
- timestamp: presentation timestamp
- video_frame_num: frame index in distorted file
- video_frame_num_expected: expected frame based on timestamp
- video_frame_num_read: VFT code value read
- video_delta_frames_N: difference from mode N
- audio_sample_num: audio sample index at beep
- audio_correlation: correlation strength
