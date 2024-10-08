#! /usr/bin/env python3


import cv2
import argparse
import time
import os


coords = []
clicked_coords = []
freeze_tags = False
use_cache_enabled = False
tag_name = ".tag_freeze"


def clear_cache():
    if os.path.exists(tag_name):
        os.remove(tag_name)


def use_cache():
    global use_cache_enabled
    use_cache_enabled = True


def are_tags_frozen():
    # no new tags will be looked at i.e. the setup is not moving
    global freeze_tags
    return freeze_tags


def mouse_callback(event, x, y, flags, param):
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_coords = []


def tag_frame(frame):
    global coords, clicked_coords, freeze_tags

    name = "Tag coordinates"

    clicked_coords = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.startWindowThread()
    cv2.namedWindow(name)
    cv2.imshow(name, frame)
    cv2.setMouseCallback(name, mouse_callback)

    while len(clicked_coords) < 4:
        key = cv2.waitKey(1)

        if key & 0xFF == ord("c"):
            break

        if key & 0xFF == ord("f"):
            freeze_tags = True
            break

        framecopy = frame.copy()
        for coord in coords:
            cv2.circle(framecopy, coord, 5, (0, 0, 255), -1)
        for coord in clicked_coords:
            cv2.circle(framecopy, coord, 5, (0, 255, 0), -1)

        text = f"Click on tag centers, tags left: {4-len(coords)}, press 'c' to continue, 'f' to freeze coordinates"
        if len(coords) > 0:
            text = f"Click on tag centers, tags left: {4-len(coords)}, press 'c' to use previous values, 'f' to freeze coordinates"
        cv2.putText(framecopy, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(framecopy, text, (11, 31), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(name, framecopy)

    if len(clicked_coords) == 4:
        coords = clicked_coords

    cv2.destroyWindow(name)

    cv2.waitKey(1)
    # vft needs to have the order matching the order of source points
    # tl, tr, bl, br
    coords = sorted(coords, key=lambda x: x[0] + x[1])
    coords = [coords[0], coords[2], coords[1], coords[3]]

    return coords


def tag_video(video_path, width=-1, height=-1):
    global coords, freeze_tags, use_cache_enabled

    if os.path.exists(tag_name):
        # Check if there is a freeze tag saved.
        # open the .tag_freeze and read coordinates
        try:
            with open(tag_name, "r") as f:
                coords = eval(f.read())
                freeze_tags = True
                return coords
        except FileNotFoundError:
            pass
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if ret:
        coords = tag_frame(frame)

    if width > 0 and height > 0:
        # Get width and height
        swidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # scale
        wratio = width / swidth
        hratio = height / sheight

        coords = [(int(x * wratio), int(y * hratio)) for x, y in coords]

        # add support for session wide freeze
        # Write coords to .tag_freeze
        if use_cache_enabled:
            with open(tag_name, "w") as f:
                f.write(str(coords))
    cap.release()

    return coords


def main():
    parser = argparse.ArgumentParser(description="Tag coordinates of a video")
    parser.add_argument("video", type=str, help="Path to the video")
    args = parser.parse_args()
    tag_video(args.video)

    print(coords)


if __name__ == "__main__":
    main()
