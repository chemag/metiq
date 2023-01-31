#!/usr/bin/env python3

"""aruco_common.py module description."""


import cv2
import numpy as np


# use fiduciarial markers from this dictionary
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

COLOR_WHITE = (255, 255, 255)


def generate_aruco_tag(tag_size, aruco_id, border_size=0, aruco_dict_id=ARUCO_DICT_ID):
    # create the aruco tag
    tag_height = tag_size - 2 * border_size
    tag_width = tag_size - 2 * border_size
    img_tag = np.zeros((tag_height, tag_width, 1), dtype=np.uint8)
    try:
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
    except AttributeError:
        # API changed for 4.7.x
        # https://stackoverflow.com/a/74975523
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    img_tag = cv2.aruco.drawMarker(aruco_dict, aruco_id, tag_height, img_tag, 1)
    # print(f"tag {aruco_id} size: {tag_height} / {tag_size}")
    # add border if needed
    if border_size > 0:
        height = tag_size
        width = tag_size
        img_full = np.zeros((height, width, 1), dtype=np.uint8)
        # add a white rectangle
        x0 = 0
        x1 = height
        y0 = 0
        y1 = width
        pts = np.array([[x0, y0], [x0, y1 - 1], [x1 - 1, y1 - 1], [x1 - 1, y0]])
        cv2.fillPoly(img_full, pts=[pts], color=COLOR_WHITE)
        # paste the tag on top of the full image
        xpos1 = border_size
        xpos2 = xpos1 + tag_width
        ypos1 = border_size
        ypos2 = ypos1 + tag_height
        img_full[ypos1:ypos2, xpos1:xpos2] = img_tag
        return img_full
    return img_tag


def detect_aruco_tags(img, aruco_dict_id=ARUCO_DICT_ID):
    # convert input to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # locate the aruco markers
    try:
        aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )
    except AttributeError:
        # API changed for 4.7.x
        # https://stackoverflow.com/a/74975523
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _rejectedImgPoints = detector.detectMarkers(gray)
    return corners, ids
