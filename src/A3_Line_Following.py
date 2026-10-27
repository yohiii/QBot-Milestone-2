import csv
import threading
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
try:
    from quick_start_qbot_platform import QBotPlatformCSICamera, QBotPlatformDriver
except ModuleNotFoundError as exc:
    if exc.name != "quick_start_qbot_platform":
        raise
    import importlib.util

    helper_path = Path(__file__).resolve().with_name("quick_start_qbot_platform.py")
    if not helper_path.is_file():
        raise ModuleNotFoundError(
            "Place quick_start_qbot_platform.py"
        ) from exc
    spec = importlib.util.spec_from_file_location("quick_start_qbot_platform", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load QBot helper: {helper_path}")
    qbot_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qbot_api)
    QBotPlatformCSICamera = qbot_api.QBotPlatformCSICamera
    QBotPlatformDriver = qbot_api.QBotPlatformDriver


ROBOT_IP = "134.29.23.233"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 400

CAMERA_FPS = 40.0
DISPLAY_FPS = 30.0

NORMAL_LOG_DIR = Path("normal_following_results")
NORMAL_LOG_INTERVAL = 0.05  # up to 20 recorded camera samples per second

CAMERA_CENTER_OFFSET_PX = 0
LINE_GUIDE_HALF_WIDTH_PX = 200
LINE_GUIDE_EARLY_MARGIN_PX = 18
LINE_GUIDE_CORRECTION_RAMP_PX = 30.0
LINE_GUIDE_MAX_FOLLOW_CORRECTION = 0.025
LINE_GUIDE_MAX_REJOIN_CORRECTION = 0.012
LINE_GUIDE_HEADING_LOOKAHEAD = 0.12
LINE_GUIDE_MIN_FOLLOW_SPEED = 0.050
CENTER_BOTTOM_ROW = 0.94
CENTER_BOTTOM_MAX_SHIFT_PX = 70
CENTER_BOTTOM_WEIGHT = 0.82
CENTER_HOLD_ERROR = 0.025
CENTER_HOLD_HEADING = 0.045
CENTER_HOLD_RELEASE_SMOOTHING = 0.86
CENTER_OPPOSITE_HEADING_FACTOR = 0.60
CENTER_TRIM_P_GAIN = 0.05
CENTER_TRIM_I_GAIN = 0.015
CENTER_TRIM_D_GAIN = 0.16
CENTER_TRIM_MAX = 0.030
CENTER_TRIM_INTEGRAL_MAX = 0.08
CENTER_TRIM_DEADBAND = 0.012
CENTER_TRIM_HEADING_START = 0.10
CENTER_TRIM_HEADING_END = 0.25
CENTER_TRIM_ERROR_START = 0.23
CENTER_TRIM_ERROR_END = 0.35
CENTER_TRIM_MAX_DT = 0.12
CENTER_SLOWDOWN_START = 0.05
CENTER_SLOWDOWN_FULL = 0.21
CENTER_SLOWDOWN_SPEED = 0.052

CORNER_RECOVERY_SCAN_ROWS = (0.92, 0.82, 0.72, 0.62, 0.52)
CORNER_RECOVERY_MAX_ROW_SHIFT_PX = 175

DARK_CONTRAST_RATIO = 0.15
MIN_DARK_CONTRAST = 18.0
SIDE_CONTRAST = 22.0
SCAN_ROWS = (0.90, 0.60, 0.2)
SCAN_HEIGHT = 9
MIN_LINE_WIDTH = 6
MAX_LINE_WIDTH = 155
START_MAX_DISTANCE = 165
TRACK_MAX_DISTANCE = 125
MAX_ROW_SHIFT = 150
MAX_NEAR_MID_SHIFT = 105

FORWARD_SPEED = 0.07
STEERING_GAIN = 0.24
HEADING_GAIN = 0.10
QBOT_CURVE_HEADING_GAIN = 0.08
FOLLOW_ERROR_DEADBAND = 0.03
STEERING_SIGN = -1.0
MAX_TURN_RATE = 0.11
TURN_SMOOTHING = 0.38


SHARP_HEADING_START = 0.16
SHARP_HEADING_FULL = 0.37
SHARP_ERROR_START = 0.38
SHARP_ERROR_FULL = 0.65
SHARP_FORWARD_SPEED = 0.030
SHARP_MAX_TURN_RATE = 0.20
SHARP_STEERING_GAIN = 0.31
SHARP_HEADING_GAIN = 0.34
SHARP_TURN_SMOOTHING = 0.36


SHARP_TURN_STEP = 0.027
SHARP_TURN_BRAKE_STEP = 0.070
CENTER_NEAR_TOL = 0.15
CENTER_BRAKE_SMOOTHING = 0.76
ACQUIRE_FRAMES = 3
LOST_FRAMES = 2

CORNER_MIN_SIDE_PX = 45
CORNER_MAX_OTHER_PX = 48
CORNER_ROW_START = 0.50
CORNER_ROW_END = 0.94
CORNER_TRIGGER_ROW = 0.58
CORNER_APPROACH_SPEED = 0.038
CORNER_CONFIRM_FRAMES = 2
CORNER_FORWARD_SPEED = 0.003
CORNER_TURN_RATE = 0.22
CORNER_MIN_SECONDS = 1.15
CORNER_MAX_SECONDS = 8.0
CORNER_CENTER_TOL = 0.90
CORNER_HEADING_TOL = 1.08
CORNER_LOCK_FRAMES = 3
CORNER_COOLDOWN = 1.4


CORNER_REJOIN_SPEED = 0.025
CORNER_REJOIN_MAX_SECONDS = 3.80
CORNER_REJOIN_LOCK_FRAMES = 4
CORNER_REJOIN_LOST_FRAMES = 5
CORNER_SAFE_REJOIN_AFTER = 4.10

REJOIN_STEERING_GAIN = 0.32
REJOIN_HEADING_GAIN = 0.26
REJOIN_MAX_TURN_RATE = 0.17
REJOIN_SMOOTHING = 0.60
REJOIN_ALIGN_ERROR = 0.09
REJOIN_ALIGN_HEADING = 0.16
REJOIN_SOFT_ERROR = 0.20
REJOIN_SOFT_HEADING = 0.30
REJOIN_SOFT_LOCK_FRAMES = 5
REJOIN_SOFT_MIN_SECONDS = 1.0
REJOIN_FALLBACK_ERROR = 0.35
REJOIN_FALLBACK_HEADING = 0.44
REJOIN_FALLBACK_MIN_SECONDS = 3.0
POST_CORNER_SLOW_SECONDS = 0.9
POST_CORNER_MAX_SPEED = 0.042
CORNER_MIN_YAW_REJOIN = 1.08
CORNER_MAX_YAW = 1.83
CORNER_TIME_FALLBACK = 2.8
CORNER_WIDE_SEARCH_SECONDS = 2.0
CORNER_MEMORY_SECONDS = 0.70
CORNER_CONFIRM_GAP = 0.30
CORNER_MAX_BAND_HEIGHT = 50
CORNER_EARLY_MIN_YAW = 0.60            # About 34 degrees, measured by gyro
CORNER_EARLY_NO_GAP_MIN_YAW = 0.73     # About 42 degrees, stricter visual proof
CORNER_EARLY_NO_GYRO_SECONDS = 2.0     # Camera-confirmed fallback only
CORNER_EARLY_MIN_SECONDS = 0.85
CORNER_EARLY_MAX_ERROR = 0.70
CORNER_EARLY_MAX_HEADING = 0.85
CORNER_EARLY_LOCK_FRAMES = 3

REJOIN_NORMAL_LOCK_FRAMES = 3
REJOIN_NORMAL_MAX_ERROR = 0.88
REJOIN_NORMAL_MAX_HEADING = 1.10
REJOIN_VISIBLE_MAX_SECONDS = 6.0

CURVE_RECOVERY_MAX_SECONDS = 2.5
CURVE_RECOVERY_NORMAL_LOCK_FRAMES = 3
CURVE_RECOVERY_MAX_NO_LINE_FRAMES = 12

MOTOR_PERIOD = 0.01
MAX_COMMAND_AGE = 0.45
MAX_NO_REPLY = 1.00
MOTOR_STILL_TIMEOUT = 3.0
MIN_WHEEL_SPEED = 0.01
MIN_POSITION_CHANGE = 0.02
MAX_NO_FRAME_SECONDS = 0.55

WINDOW = "QBot - M: START / S: STOP / R: STEER / P: SNAP / Q: QUIT"
MASK_WINDOW = "BLACK LINE MASK - black tape appears WHITE"


def clamp(value, low, high):
    return max(low, min(high, value))


def guide_rail_assist(error, heading, forward, turn, steering_sign,
                      max_turn_rate, rejoin=False):
    
    if not (np.isfinite(error) and np.isfinite(heading)):
        return forward, turn

    image_half_width = CAMERA_WIDTH * 0.5
    projected_offset_px = image_half_width * (
        error + LINE_GUIDE_HEADING_LOOKAHEAD * heading)
    start_px = max(0.0, LINE_GUIDE_HALF_WIDTH_PX - LINE_GUIDE_EARLY_MARGIN_PX)
    outside_amount = max(0.0, abs(projected_offset_px) - start_px)
    response = clamp(outside_amount / LINE_GUIDE_CORRECTION_RAMP_PX,
                     0.0, 1.0)
    if response <= 0.0:
        return forward, turn

    max_extra = (LINE_GUIDE_MAX_REJOIN_CORRECTION if rejoin
                 else LINE_GUIDE_MAX_FOLLOW_CORRECTION)
    correction = (steering_sign * np.sign(projected_offset_px) *
                  max_extra * response)
    corrected_turn = clamp(turn + correction, -max_turn_rate, max_turn_rate)
    if rejoin:
        
        return forward, corrected_turn

    allowed_speed = (FORWARD_SPEED -
                     (FORWARD_SPEED - LINE_GUIDE_MIN_FOLLOW_SPEED) * response)
    return min(forward, allowed_speed), corrected_turn


def mask_segments(band):
    b = (np.asarray(band) > 0).astype(np.int8)
    diff = np.diff(np.pad(b, (1, 1)))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def make_black_mask(frame):
    """Build a black-line mask using row-local background brightness."""
    if frame is None or frame.size == 0:
        raise ValueError("Camera returned empty frame")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)
    gray = cv2.medianBlur(gray, 3)
    brightness = np.percentile(gray, 75, axis=1).astype(np.float32)
    min_difference = np.maximum(MIN_DARK_CONTRAST, DARK_CONTRAST_RATIO * brightness)
    threshold = np.maximum(0.0, brightness - min_difference)
    mask = (gray.astype(np.float32) < threshold[:, None]).astype(np.uint8) * 255
    return gray, mask, brightness


def scan_band(gray, mask, row_fraction, expected_x, max_distance):
    """Find a narrow dark stripe with a brighter background on both sides."""
    h, w = mask.shape
    y = int(clamp(round((h - 1) * row_fraction), 0, h - 1))
    half = SCAN_HEIGHT // 2
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    binary_band = np.median(mask[y0:y1], axis=0) > 127
    gray_band = np.median(gray[y0:y1], axis=0)
    candidates = []
    for x0, x1 in mask_segments(binary_band):
        width = x1 - x0
        if not MIN_LINE_WIDTH <= width <= MAX_LINE_WIDTH:
            continue
        center = (x0 + x1 - 1) / 2.0
        offset = abs(center - expected_x)
        if offset > max_distance:
            continue

        side = int(clamp(width * 0.40, 9, 25))
        if x0 < side or x1 + side > w:
            continue
        inside_gray = float(np.median(gray_band[x0:x1]))
        left_gray = float(np.median(gray_band[x0-side:x0]))
        right_gray = float(np.median(gray_band[x1:x1+side]))
        contrast = min(left_gray, right_gray) - inside_gray
        if contrast < SIDE_CONTRAST:
            continue

        score = offset - 0.06 * contrast
        candidates.append((score, center, width))
    if not candidates:
        return None, y, 0
    _, x, width = min(candidates)
    return x, y, width



def inspect_frame(frame, previous_x=None, wide=False, prepared=None, render=True):
    
    if prepared is None:
        gray, mask, _ = make_black_mask(frame)
    else:
        gray, mask = prepared
    h, w = mask.shape
    center_x = w/2.0 + CAMERA_CENTER_OFFSET_PX
    expected = center_x if previous_x is None else previous_x


    fractions = CORNER_RECOVERY_SCAN_ROWS if wide else SCAN_ROWS
    points = []
    for i, fraction in enumerate(fractions):
        if wide:
            max_distance = int(w * (0.46 if i == 0 else 0.36))
        else:
            max_distance = ((START_MAX_DISTANCE if previous_x is None
                             else TRACK_MAX_DISTANCE) if i == 0 else MAX_ROW_SHIFT)
        x, y, line_width = scan_band(gray, mask, fraction, expected, max_distance)
        points.append((x, y, line_width))
        if x is not None:
            expected = x

    if wide:
        
        pair = None
        for i in range(len(points) - 1):
            near_x, far_x = points[i][0], points[i + 1][0]
            if (near_x is not None and far_x is not None and
                    abs(near_x - far_x) <= CORNER_RECOVERY_MAX_ROW_SHIFT_PX):
                pair = (i, near_x, far_x)
                break
        reliable = pair is not None
        if reliable:
            i, near_x, far_x = pair
            track_x = 0.70 * near_x + 0.30 * far_x
            heading = (far_x - near_x) / (w / 2.0)
            anchor_x = near_x
        else:
            track_x, heading, anchor_x = center_x, 0.0, None
    else:

        near_x, _, _ = points[0]
        mid_x, _, _ = points[1]
        far_x, _, _ = points[2]
        valid_near = near_x is not None and mid_x is not None and abs(near_x-mid_x) <= MAX_NEAR_MID_SHIFT
        valid_mid = mid_x is not None and far_x is not None and abs(mid_x-far_x) <= MAX_ROW_SHIFT
        reliable = valid_near or (near_x is None and valid_mid)

        if valid_near:

            track_x = 0.90*near_x + 0.10*mid_x
            bottom_x, bottom_y, _ = scan_band(
                gray, mask, CENTER_BOTTOM_ROW, near_x, CENTER_BOTTOM_MAX_SHIFT_PX)
            if (bottom_x is not None and
                    abs(bottom_x - near_x) <= CENTER_BOTTOM_MAX_SHIFT_PX):
                track_x = (CENTER_BOTTOM_WEIGHT * bottom_x +
                           (1.0 - CENTER_BOTTOM_WEIGHT) * near_x)
            heading = ((far_x if far_x is not None else mid_x) - near_x) / (w/2.0)
            anchor_x = near_x
        elif reliable:
            track_x = 0.70*mid_x + 0.30*far_x
            heading = (far_x-mid_x)/(w/2.0)
            anchor_x = mid_x
        else:
            track_x, heading, anchor_x = w/2.0, 0.0, None


    error = (track_x-center_x)/(w/2.0) if reliable else 0.0
    view = None
    if render:
        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        left_rail = int(round(clamp(center_x - LINE_GUIDE_HALF_WIDTH_PX,
                                    0, w - 1)))
        right_rail = int(round(clamp(center_x + LINE_GUIDE_HALF_WIDTH_PX,
                                     0, w - 1)))
        cv2.line(view, (left_rail, 0), (left_rail, h - 1),
                 (0, 220, 0), 2)
        cv2.line(view, (right_rail, 0), (right_rail, h - 1),
                 (0, 220, 0), 2)
        
        cv2.line(view, (int(round(center_x)), 0), (int(round(center_x)), h - 1),
                 (255, 120, 0), 2)
        for x, y, line_width in points:
            cv2.line(view, (0, y), (w - 1, y), (95, 95, 95), 1)
            if x is not None:
                cv2.circle(view, (int(round(x)), y), 3,
                           (0, 220, 0) if reliable else (0, 0, 255), -1)
        if reliable:
            if not wide and valid_near and bottom_x is not None:
                cv2.circle(view, (int(round(bottom_x)), int(bottom_y)), 3,
                           (255, 255, 0), 2)
          
            cv2.circle(view, (int(round(track_x)), points[0][1]), 5,
                       (255, 0, 255), 2)
            cv2.putText(view, "CENTER GAP {:+d}px".format(int(round(track_x - center_x))),
                        (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2)
            
            in_guide = abs(track_x - center_x) <= LINE_GUIDE_HALF_WIDTH_PX
            cv2.putText(view,
                        "GUIDE: {} (center +/-{}px)".format(
                            "IN" if in_guide else "OUT", LINE_GUIDE_HALF_WIDTH_PX),
                        (8, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 220, 0) if in_guide else (0, 120, 255), 2)
        cv2.putText(view, "BLACK LINE: {} | err={:+.2f} heading={:+.2f}".format(
            "YES" if reliable else "NO", error, heading), (8, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        cv2.putText(view, "W: {} / {} / {} px".format(*(p[2] for p in points[:3])),
                    (8, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (0, 255, 255), 2)
    return reliable, error, heading, anchor_x, view, mask


def detect_sharp_corner(mask, anchor_x):
   
    if anchor_x is None:
        return None, None
    h, w = mask.shape
    x = int(round(anchor_x))
    if not 35 <= x < w - 35:
        return None, None
    start_y = max(8, int(h * CORNER_ROW_START))
    stop_y = min(h - 13, int(h * CORNER_ROW_END))
    candidates = []

    for y in range(start_y, stop_y + 1, 3):
        band = (np.mean(mask[y-2:y+3] > 127, axis=0) >= .6)
        for x0, x1 in mask_segments(band):
            if x0 > x + 25 or x1 < x - 25:
                continue
            
            attach = int(clamp(x, x0, x1-1))
            left_len = attach - x0
            right_len = x1 - attach
            if (left_len >= CORNER_MIN_SIDE_PX and
                    right_len <= CORNER_MAX_OTHER_PX):
                direction, side_len = "LEFT", left_len
                probe = int(attach - min(55, .60*left_len))
            elif (right_len >= CORNER_MIN_SIDE_PX and
                  left_len <= CORNER_MAX_OTHER_PX):
                direction, side_len = "RIGHT", right_len
                probe = int(attach + min(55, .60*right_len))
            else:
                continue
            if not 10 <= probe < w-10:
                continue

            
            above, below = None, None
            for margin in (19, 28, 38, 49):
                if y-margin-3 < 0 or y+margin+3 >= h:
                    continue
                a = float(np.mean(mask[y-margin-3:y-margin+4,
                                       probe-5:probe+6] > 127))
                b = float(np.mean(mask[y+margin-3:y+margin+4,
                                       probe-5:probe+6] > 127))
                if a < .14 and b < .14:
                    above, below = a, b
                    break
            if above is None:
                continue

            
            if y-46 >= 4:
                long_ahead = np.mean(mask[y-46:y-34,
                                          max(0, attach-11):min(w, attach+12)] > 127)
                if long_ahead > .40:
                    continue
            candidates.append((direction, y, side_len))
            break

    for direction in ('LEFT', 'RIGHT'):
        rows = sorted(y for d, y, _ in candidates if d == direction)
        opposite = [y for d, y, _ in candidates if d != direction]
        if len(rows) < 2 or len(opposite) >= 2:
            continue
        runs = [[rows[0]]]
        for y in rows[1:]:
            if y - runs[-1][-1] <= 6:
                runs[-1].append(y)
            else:
                runs.append([y])
        runs.sort(key=lambda r: (len(r), np.median(r)), reverse=True)
        if len(runs[0]) >= 2:
            return direction, int(np.median(runs[0]))
    return None, None


def follow_command(error, heading, previous_turn, steering_sign=STEERING_SIGN, center_trim=0.0):
    
    tight_by_heading = clamp(
        (abs(heading) - SHARP_HEADING_START) /
        (SHARP_HEADING_FULL - SHARP_HEADING_START), 0.0, 1.0)
    tight_by_error = clamp(
        (abs(error) - SHARP_ERROR_START) /
        (SHARP_ERROR_FULL - SHARP_ERROR_START), 0.0, 1.0)
    tight = max(tight_by_heading, tight_by_error)


    gain_error = STEERING_GAIN + tight * (SHARP_STEERING_GAIN - STEERING_GAIN)
    gain_heading = HEADING_GAIN + tight * (SHARP_HEADING_GAIN - HEADING_GAIN)
    turn_limit = MAX_TURN_RATE + tight * (SHARP_MAX_TURN_RATE - MAX_TURN_RATE)
    smoothing = TURN_SMOOTHING + tight * (SHARP_TURN_SMOOTHING - TURN_SMOOTHING)
    forward = FORWARD_SPEED + tight * (SHARP_FORWARD_SPEED - FORWARD_SPEED)


    heading_weight = 0.32 + 0.68 * clamp(
        (abs(error) - 0.06) / (CENTER_NEAR_TOL - 0.06), 0.0, 1.0)
   
    if abs(error) < CENTER_NEAR_TOL and error * heading < 0.0:
        heading_weight *= CENTER_OPPOSITE_HEADING_FACTOR
    target_turn = steering_sign * (gain_error * error +
                                   gain_heading * heading_weight * heading)
    center_hold = (abs(error) <= CENTER_HOLD_ERROR and
                   abs(heading) <= CENTER_HOLD_HEADING)
    if center_hold:
       
        target_turn = 0.0
    
    target_turn += center_trim
    target_turn = clamp(target_turn, -turn_limit, turn_limit)
    releasing = (abs(error) <= CENTER_NEAR_TOL and
                 (abs(target_turn) < abs(previous_turn) or
                  target_turn * previous_turn < 0.0))
    response = (CENTER_HOLD_RELEASE_SMOOTHING if center_hold else
                max(smoothing, CENTER_BRAKE_SMOOTHING) if releasing else smoothing)
    turn = response * target_turn + (1.0 - response) * previous_turn
    if tight > 0.01:


        step = (SHARP_TURN_BRAKE_STEP if abs(turn) < abs(previous_turn)
                else SHARP_TURN_STEP)
        turn = clamp(turn, previous_turn - step, previous_turn + step)
    return forward, clamp(turn, -turn_limit, turn_limit)


def qbot_curve_command(error, heading, previous_turn, steering_sign=STEERING_SIGN):
    
    tight_by_heading = clamp(
        (abs(heading) - SHARP_HEADING_START) /
        (SHARP_HEADING_FULL - SHARP_HEADING_START), 0.0, 1.0)
    tight_by_error = clamp(
        (abs(error) - SHARP_ERROR_START) /
        (SHARP_ERROR_FULL - SHARP_ERROR_START), 0.0, 1.0)
    tight = max(tight_by_heading, tight_by_error)

   
    gain_error = STEERING_GAIN + tight * (SHARP_STEERING_GAIN - STEERING_GAIN)
    gain_heading = QBOT_CURVE_HEADING_GAIN + tight * (SHARP_HEADING_GAIN - QBOT_CURVE_HEADING_GAIN)
    turn_limit = MAX_TURN_RATE + tight * (SHARP_MAX_TURN_RATE - MAX_TURN_RATE)
    smoothing = TURN_SMOOTHING + tight * (SHARP_TURN_SMOOTHING - TURN_SMOOTHING)
    forward = FORWARD_SPEED + tight * (SHARP_FORWARD_SPEED - FORWARD_SPEED)

   
    if abs(error) <= FOLLOW_ERROR_DEADBAND:
        corrected_error = 0.0
    else:
        corrected_error = np.sign(error) * (abs(error) - FOLLOW_ERROR_DEADBAND)

    target_turn = steering_sign * (
        gain_error * corrected_error + gain_heading * heading
    )
    target_turn = clamp(target_turn, -turn_limit, turn_limit)
    turn = smoothing * target_turn + (1.0 - smoothing) * previous_turn
    if tight > 0.01:
        
        step = (SHARP_TURN_BRAKE_STEP if abs(turn) < abs(previous_turn)
                else SHARP_TURN_STEP)
        turn = clamp(turn, previous_turn - step, previous_turn + step)
    return forward, clamp(turn, -turn_limit, turn_limit)


def regular_line_is_trackable(found, error, heading):
   
    return (bool(found) and np.isfinite(error) and np.isfinite(heading) and
            abs(error) <= REJOIN_NORMAL_MAX_ERROR and
            abs(heading) <= REJOIN_NORMAL_MAX_HEADING)


def early_corner_line_is_trackable(found, error, heading):
    """Use stricter geometry when releasing an explicit corner turn early."""
    return (bool(found) and np.isfinite(error) and np.isfinite(heading) and
            abs(error) <= CORNER_EARLY_MAX_ERROR and
            abs(heading) <= CORNER_EARLY_MAX_HEADING)


def corner_early_reacquire_allowed(elapsed, yaw_progress, gap_seen,
                                    normal_found, normal_error, normal_heading):
    
    if elapsed < CORNER_EARLY_MIN_SECONDS:
        return False
    if not early_corner_line_is_trackable(
            normal_found, normal_error, normal_heading):
        return False
    rotated = (yaw_progress >= CORNER_EARLY_MIN_YAW
               if yaw_progress is not None
               else elapsed >= CORNER_EARLY_NO_GYRO_SECONDS)
    if not rotated:
        return False
    if gap_seen:
        return True
    
    return (yaw_progress is not None and
            yaw_progress >= CORNER_EARLY_NO_GAP_MIN_YAW and
            abs(normal_error) <= 0.30 and abs(normal_heading) <= 0.35)


def rejoin_command(error, heading, previous_turn, steering_sign=STEERING_SIGN,
                   use_guide=False):
    
    target = steering_sign * (REJOIN_STEERING_GAIN * error + REJOIN_HEADING_GAIN * heading)
    target = clamp(target, -REJOIN_MAX_TURN_RATE, REJOIN_MAX_TURN_RATE)
    releasing = (target * previous_turn <= 0.0 or abs(target) < abs(previous_turn))
    response = 0.82 if releasing else REJOIN_SMOOTHING
    turn = response * target + (1.0 - response) * previous_turn
    forward = CORNER_REJOIN_SPEED
    turn = clamp(turn, -REJOIN_MAX_TURN_RATE, REJOIN_MAX_TURN_RATE)
    if use_guide:
        forward, turn = guide_rail_assist(
            error, heading, forward, turn, steering_sign,
            REJOIN_MAX_TURN_RATE, rejoin=True)
    return forward, turn


def rejoin_alignment_status(found, error, heading):
    
    if not regular_line_is_trackable(found, error, heading):
        return False, False
    precise = abs(error) <= REJOIN_ALIGN_ERROR and abs(heading) <= REJOIN_ALIGN_HEADING
    soft = abs(error) <= REJOIN_SOFT_ERROR and abs(heading) <= REJOIN_SOFT_HEADING
    return precise, soft


class LineCenteringTrim:
   
    def __init__(self):
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.rate = 0.0
        self.previous_error = None
        self.last_update = None
        self.last_trim = 0.0

    def update(self, error, heading, steering_sign=STEERING_SIGN, now=None):
        if not np.isfinite(error) or not np.isfinite(heading):
            self.reset()
            return 0.0
        if now is None:
            now = time.monotonic()
        dt = (1.0 / CAMERA_FPS if self.last_update is None else
              clamp(now - self.last_update, 0.001, CENTER_TRIM_MAX_DT))
        self.last_update = now

       
        heading_factor = 1.0 - clamp(
            (abs(heading) - CENTER_TRIM_HEADING_START) /
            (CENTER_TRIM_HEADING_END - CENTER_TRIM_HEADING_START), 0.0, 1.0)
        error_factor = 1.0 - clamp(
            (abs(error) - CENTER_TRIM_ERROR_START) /
            (CENTER_TRIM_ERROR_END - CENTER_TRIM_ERROR_START), 0.0, 1.0)
        activity = heading_factor * error_factor

        if self.previous_error is None:
            raw_rate = 0.0
        else:
            raw_rate = clamp((error - self.previous_error) / dt, -1.0, 1.0)
            if (abs(error) > CENTER_TRIM_DEADBAND and
                    abs(self.previous_error) > CENTER_TRIM_DEADBAND and
                    error * self.previous_error < 0.0):
                self.integral *= 0.25
        self.previous_error = error
        self.rate = 0.78 * self.rate + 0.22 * raw_rate


        self.integral *= max(0.0, 1.0 - 0.25 * dt)
        if abs(error) > CENTER_TRIM_DEADBAND and activity > 0.10:
            self.integral += error * dt * activity
        else:
            self.integral *= max(0.0, 1.0 - 1.5 * dt)
        self.integral = clamp(self.integral,
                              -CENTER_TRIM_INTEGRAL_MAX,
                              CENTER_TRIM_INTEGRAL_MAX)

        trim = steering_sign * activity * (
            CENTER_TRIM_P_GAIN * error +
            CENTER_TRIM_I_GAIN * self.integral +
            CENTER_TRIM_D_GAIN * self.rate)
        self.last_trim = clamp(trim, -CENTER_TRIM_MAX, CENTER_TRIM_MAX)
        return self.last_trim


def centering_speed_limit(error, heading):
    
    heading_factor = 1.0 - clamp(
        (abs(heading) - CENTER_TRIM_HEADING_START) /
        (CENTER_TRIM_HEADING_END - CENTER_TRIM_HEADING_START), 0.0, 1.0)
    offset_factor = clamp(
        (abs(error) - CENTER_SLOWDOWN_START) /
        (CENTER_SLOWDOWN_FULL - CENTER_SLOWDOWN_START), 0.0, 1.0)
    return FORWARD_SPEED - (FORWARD_SPEED - CENTER_SLOWDOWN_SPEED) * heading_factor * offset_factor


def hybrid_follow_command(error, heading, previous_turn,
                          center_controller, steering_sign=STEERING_SIGN):
    
    use_qbot_curve = (abs(heading) >= SHARP_HEADING_START or
                      abs(error) >= SHARP_ERROR_START)
    if use_qbot_curve:
        center_controller.reset()
        forward, turn = qbot_curve_command(
            error, heading, previous_turn, steering_sign)
    else:
        trim = center_controller.update(error, heading, steering_sign)
        forward, turn = follow_command(
            error, heading, previous_turn, steering_sign, center_trim=trim)
        forward = min(forward, centering_speed_limit(error, heading))
    # Keep the File 38/QBot_line command intact near center. Apply a small
    # corridor correction only when the tape is approaching either green rail.
    tight = max(
        clamp((abs(heading) - SHARP_HEADING_START) /
              (SHARP_HEADING_FULL - SHARP_HEADING_START), 0.0, 1.0),
        clamp((abs(error) - SHARP_ERROR_START) /
              (SHARP_ERROR_FULL - SHARP_ERROR_START), 0.0, 1.0))
    turn_limit = MAX_TURN_RATE + tight * (SHARP_MAX_TURN_RATE - MAX_TURN_RATE)
    forward, turn = guide_rail_assist(
        error, heading, forward, turn, steering_sign, turn_limit)
    return forward, turn, use_qbot_curve


class NormalLineCSV:
    """Record camera measurements and requested motor commands for one run."""

    HEADER = (
        "time_s", "state", "line_detected", "error_norm", "error_px",
        "heading_norm", "forward_cmd_mps", "turn_cmd_radps",
        "trim_cmd_radps", "curve_mode", "scan_mode", "camera_fps",
        "steering_sign"
    )

    def __init__(self):
        self._file = None
        self._writer = None
        self._start_time = None
        self._last_record = None
        self.path = None
        self.samples = 0

    @property
    def active(self):
        return self._file is not None

    def start(self):
        if self.active:
            self.close()
        NORMAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        filename = "normal_line_following_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.path = NORMAL_LOG_DIR / filename
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._file.flush()
        self._start_time = time.monotonic()
        self._last_record = None
        self.samples = 0
        print("[CSV] Recording:", self.path)

    def record(self, state, found, error, heading, forward, turn,
               trim, curve_mode, wide_scan, camera_fps, steering_sign):
        if not self.active:
            return
        now = time.monotonic()
        if (self._last_record is not None and
                now - self._last_record < NORMAL_LOG_INTERVAL):
            return
        self._last_record = now
        # A lost line is missing data, not an observed zero tracking error.
        valid = bool(found) and np.isfinite(error) and np.isfinite(heading)
        self._writer.writerow((
            round(now - self._start_time, 4),
            state,
            int(valid),
            round(float(error), 6) if valid else "",
            round(float(error) * CAMERA_WIDTH / 2.0, 3) if valid else "",
            round(float(heading), 6) if valid else "",
            round(float(forward), 6),
            round(float(turn), 6),
            round(float(trim), 6),
            int(bool(curve_mode)),
            "WIDE" if wide_scan else "NORMAL",
            round(float(camera_fps), 2),
            float(steering_sign),
        ))
        self.samples += 1
       
        self._file.flush()

    def close(self):
        if self._file is None:
            return
        try:
            self._file.flush()
            self._file.close()
        finally:
            self._file = None
            self._writer = None
        print("[CSV] Saved {} samples -> {}".format(self.samples, self.path))


class ContinuousMotor:
    

    def __init__(self, ip):
        self.ip = ip
        self._lock = threading.Lock()
        self._target = (0.0, 0.0)
        self._updated = time.monotonic()
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="QBotMotor", daemon=True)
        self.connected = False
        self.error = None
        self._yaw_total = 0.0
        self._yaw_sample_count = 0
        self._last_yaw_time = None

    def yaw_snapshot(self):
        """(integrated yaw radians, count of valid moving gyro samples)."""
        with self._lock:
            return self._yaw_total, self._yaw_sample_count

    def yaw_since(self, snapshot):
        with self._lock:
            theta, samples = snapshot


            if self._yaw_sample_count - samples < 5:
                return None
            delta = abs(self._yaw_total - theta)

            if delta < 0.15:
                return None
            return delta

    def start(self):
        self._thread.start()
        if not self._ready.wait(timeout=4.0):
            self.stop()
            raise RuntimeError("Motor connection timed out")
        if not self.connected:
            raise RuntimeError("Cannot connect QBot Driver at {}:18888 ({})".format(
                self.ip, self.error))

    def command(self, forward, turn):
        with self._lock:
            self._target = (float(forward), float(turn))
            self._updated = time.monotonic()

    def command_snapshot(self):
        """Thread-safe copy of requested speed and turn (not measured speeds)."""
        with self._lock:
            return self._target

    def stop(self):
        self.command(0.0, 0.0)

    def close(self):
        self.stop()
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print("WARNING: motor thread has not exited; use physical emergency stop")

    def _run(self):
        driver = None
        last_reply = time.monotonic()
        still_since = None
        reference_position = None
        try:
            driver = QBotPlatformDriver(ip=self.ip, mode=1)
            if not driver._handle.connected:
                raise RuntimeError("Driver not connected")
            self.connected = True
            self._ready.set()
            print("MOTOR DRIVER CONNECTED; motor thread is sending commands continuously")
            next_update = time.monotonic()
            while not self._shutdown.is_set():
                now = time.monotonic()
                with self._lock:
                    forward, turn = self._target
                    age = now - self._updated
                if age > MAX_COMMAND_AGE:
                    forward, turn = 0.0, 0.0

                replied = bool(driver.read_write_std(
                    timestamp=time.time(), arm=1,
                    commands=np.array([forward, turn], dtype=np.float64)))
                if not driver._handle.connected:
                    raise RuntimeError("QBot driver connection lost")
                if replied:
                    stamp = time.monotonic()
                    last_reply = stamp


                    gyro_z = float(np.asarray(driver.gyroscope)[2])
                    if abs(turn) >= 0.06 and np.isfinite(gyro_z) and 0.035 <= abs(gyro_z) <= 2.0:
                        with self._lock:
                            if self._last_yaw_time is not None:
                                dt = stamp - self._last_yaw_time
                                if 0.0 < dt < 0.12:
                                    self._yaw_total += gyro_z * dt
                            self._yaw_sample_count += 1
                    self._last_yaw_time = stamp
                if time.monotonic() - last_reply > MAX_NO_REPLY:
                    raise RuntimeError("No fresh driver reply for {:.1f} sec".format(MAX_NO_REPLY))

                moving_cmd = abs(forward) > 0.02 or abs(turn) > 0.02
                if moving_cmd and replied:
                    speeds = np.asarray(driver.wheelSpeeds, dtype=float)
                    positions = np.asarray(driver.wheelPositions, dtype=float).copy()
                    if reference_position is None:
                        reference_position = positions
                    observed = (np.any(np.abs(speeds) > MIN_WHEEL_SPEED) or
                                np.any(np.abs(positions - reference_position) > MIN_POSITION_CHANGE))
                    if observed:
                        still_since = None
                        reference_position = positions
                    elif still_since is None:
                        still_since = time.monotonic()
                    if still_since is not None and time.monotonic() - still_since > MOTOR_STILL_TIMEOUT:
                        raise RuntimeError(
                            "MOTOR NOT MOVING: driver replies, wheel speeds/positions stay zero")
                else:
                    still_since = None
                    reference_position = None

                next_update += MOTOR_PERIOD
                sleep_time = next_update - time.monotonic()
                if sleep_time > 0:
                    self._shutdown.wait(sleep_time)
                else:

                    next_update = time.monotonic()
        except Exception as exc:
            self.error = str(exc)
            print("MOTOR THREAD ERROR:", self.error)
        finally:
            self.connected = False
            self._ready.set()
            if driver is not None:
                try:
                    for _ in range(8):
                        if not driver._handle.connected:
                            break
                        driver.read_write_std(
                            timestamp=time.time(), arm=1,
                            commands=np.array([0.0, 0.0], dtype=np.float64))
                        time.sleep(0.01)
                except Exception as exc:
                    print("WARNING: stop command not confirmed:", exc)
                try:
                    driver.terminate()
                except Exception as exc:
                    print("WARNING: driver terminate failed:", exc)
            print("MOTOR THREAD EXITED: verify wheels are stationary")


def main():
    camera = None
    motor = None
    state = "PREVIEW"
    previous_x = None
    acquire_frames = 0
    lost_frames = 0
    last_steer = 0.0
    steering_sign = STEERING_SIGN
    center_controller = LineCenteringTrim()
    curve_active = False
    corner_candidate = None
    corner_frames = 0
    corner_seen_at = -100.0
    corner_seen_y = -1
    corner_begin = 0.0
    corner_yaw_start = None

    corner_lock = 0
    corner_early_lock = 0
    corner_had_gap = False
    corner_gap_frames = 0
    rejoin_begin = 0.0
    rejoin_lock = 0
    rejoin_normal_lock = 0
    rejoin_align_lock = 0
    rejoin_soft_lock = 0
    rejoin_lost = 0
    follow_slow_until = 0.0
    last_display_time = -100.0
    measured_vision_hz = 0.0
    previous_vision_time = None
    curve_recovery_begin = 0.0
    curve_recovery_lock = 0
    curve_recovery_lost = 0
    last_corner_end = -100.0
    last_tracked_at = -100.0
    last_track_anchor = None
    last_frame_time = time.monotonic()
    view = None
    mask = None
    normal_csv = NormalLineCSV()
   
    print("Robot IP:", ROBOT_IP)
    print("QBot: M=start+record | S=stop+save CSV | R=reverse | P=snapshot | Q=quit")

    try:
        camera = QBotPlatformCSICamera(
            frameWidth=CAMERA_WIDTH, frameHeight=CAMERA_HEIGHT,
            frameRate=CAMERA_FPS)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

        while True:
            if state != "FOLLOW":
                center_controller.reset()
            has_frame = bool(camera.read())
            found = False
            error = heading = 0.0
            normal_found = False
            normal_error = normal_heading = 0.0
            normal_x = None
            wide_state = False
            corner_dir = None
            corner_y = None
            if has_frame:
                frame_time = time.monotonic()
                last_frame_time = frame_time
                if previous_vision_time is not None:
                    dt = frame_time - previous_vision_time
                    if dt > 0.001:
                        instant_rate = 1.0 / dt
                        measured_vision_hz = (instant_rate if measured_vision_hz == 0.0
                                              else 0.85 * measured_vision_hz + 0.15 * instant_rate)
                previous_vision_time = frame_time

                turn_state = state in ("TURN_LEFT", "TURN_RIGHT")
                wide_state = (turn_state or state.startswith("REJOIN_") or
                              state == "RECOVER_CURVE" or
                              (state == "WAIT_LINE" and
                               frame_time - last_corner_end <= CORNER_WIDE_SEARCH_SECONDS))

                
                gray, mask, _ = make_black_mask(camera.imageData)
                display_due = frame_time - last_display_time >= 1.0 / DISPLAY_FPS
                found, error, heading, new_x, new_view, mask = inspect_frame(
                    camera.imageData, None if wide_state else previous_x,
                    wide=wide_state, prepared=(gray, mask), render=display_due)
                if new_view is not None:
                    view = new_view

                if state in ("TURN_LEFT", "TURN_RIGHT", "REJOIN_LEFT",
                             "REJOIN_RIGHT", "RECOVER_CURVE") or (state == "WAIT_LINE" and wide_state):
                    normal_found, normal_error, normal_heading, normal_x, _, _ = inspect_frame(
                        camera.imageData, None, wide=False,
                        prepared=(gray, mask), render=False)
                if found:
                    if not turn_state:
                        previous_x = new_x
                    if state in ("FOLLOW", "WAIT_LINE"):
                        last_tracked_at = frame_time
                        last_track_anchor = new_x

                recent_track = frame_time - last_tracked_at <= CORNER_MEMORY_SECONDS
                if state in ("FOLLOW", "WAIT_LINE") and recent_track:
                    corner_dir, corner_y = detect_sharp_corner(mask, last_track_anchor)

                if display_due and view is not None:
                    cv2.putText(view, "STATE: " + state, (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, .75,
                                (0, 255, 0) if state == "FOLLOW" else (0, 150, 255), 2)
                    cv2.putText(view, "STEER: {:+.0f} | CAMERA: {:.1f}Hz".format(
                        steering_sign, measured_vision_hz), (8, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 2)
                    cv2.putText(view, "CORNER: {} (confirm {}/{})".format(
                        corner_dir or "NONE", corner_frames, CORNER_CONFIRM_FRAMES),
                        (8, 127), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 2)
                    if state.startswith("TURN_") or state.startswith("REJOIN_"):
                        cv2.putText(view, "TURN {:.1f}s | normal lock {}".format(
                            frame_time - corner_begin, rejoin_normal_lock), (8, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 190, 255), 2)
                    if state.startswith("REJOIN_"):
                        cv2.putText(view, "ALIGN {} | SOFT {} | target +/-{:.2f}".format(
                            rejoin_align_lock, rejoin_soft_lock, REJOIN_ALIGN_ERROR),
                            (8, 173), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 200, 255), 2)
                    if corner_y is not None:
                        cv2.line(view, (0, int(corner_y)), (CAMERA_WIDTH - 1, int(corner_y)),
                                 (255, 0, 255), 1)
                    cv2.imshow(WINDOW, view)
                    cv2.imshow(MASK_WINDOW, mask)
                    last_display_time = frame_time

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("QUIT")
                break
            if key in (ord('s'), ord('S')):
                if motor is not None:
                    motor.stop()
                normal_csv.close()
                state = "PREVIEW"
                previous_x = None
                last_steer = 0.0
                acquire_frames = lost_frames = 0
                corner_candidate = None
                corner_frames = 0
                corner_lock = 0
                corner_early_lock = 0
                corner_had_gap = False
                corner_gap_frames = 0
                rejoin_lock = rejoin_lost = 0
                rejoin_normal_lock = 0
                curve_recovery_lock = curve_recovery_lost = 0
                corner_seen_y = -1
                corner_seen_at = -100.0
                last_tracked_at = -100.0
                last_track_anchor = None
                print("S -> STOP; press M to restart")
                continue
            if key in (ord('r'), ord('R')):

                if motor is not None:
                    motor.stop()
                normal_csv.close()
                state = "PREVIEW"
                previous_x = None
                last_steer = 0.0
                acquire_frames = lost_frames = 0
                corner_candidate = None
                corner_frames = 0
                corner_lock = 0
                corner_early_lock = 0
                corner_had_gap = False
                corner_gap_frames = 0
                rejoin_lock = rejoin_lost = 0
                rejoin_normal_lock = 0
                curve_recovery_lock = curve_recovery_lost = 0
                corner_seen_y = -1
                corner_seen_at = -100.0
                last_tracked_at = -100.0
                last_track_anchor = None
                steering_sign *= -1.0
                print("STEERING DIRECTION REVERSED -> SIGN={:+.0f}. Press M to restart.".format(steering_sign))
                continue
            if key in (ord('p'), ord('P')) and view is not None and mask is not None:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                image_path = Path('qbot_camera_{}.png'.format(stamp))
                mask_path = Path('qbot_mask_{}.png'.format(stamp))
                cv2.imwrite(str(image_path), view)
                cv2.imwrite(str(mask_path), mask)
                print("SAVED:", image_path, mask_path)

            if key in (ord('m'), ord('M')) and state == "PREVIEW":
                if motor is not None and not motor.connected:
                    motor.close()
                    motor = None
                if motor is None:
                    motor = ContinuousMotor(ROBOT_IP)
                    try:
                        motor.start()
                    except Exception as exc:
                        print("CANNOT START MOTOR:", exc)
                        motor.close()
                        motor = None
                if motor is not None and motor.connected:
                    last_frame_time = time.monotonic()
                    state = "WAIT_LINE"
                    previous_x = None
                    last_steer = 0.0
                    acquire_frames = lost_frames = 0
                    corner_candidate = None
                    corner_frames = 0
                    corner_lock = 0
                    corner_early_lock = 0
                    rejoin_normal_lock = 0
                    curve_recovery_lock = 0
                    curve_recovery_lost = 0
                    corner_had_gap = False
                    corner_gap_frames = 0
                    corner_seen_y = -1
                    corner_seen_at = -100.0
                    last_tracked_at = -100.0
                    last_track_anchor = None
                    try:
                        normal_csv.start()
                    except OSError as exc:
                        motor.stop()
                        state = "PREVIEW"
                        print("[CSV] Cannot record; robot stopped:", exc)
                        continue
                    print("M -> WAIT_LINE: release M; searching for the black line")

            if state == "PREVIEW":
                if motor is not None:
                    motor.stop()
                time.sleep(0.005)
                continue

            if motor is None or not motor.connected:
                print("MOTOR DISCONNECTED:", motor.error if motor is not None else "unknown")
                normal_csv.close()
                if motor is not None:
                    motor.stop()
                state = "PREVIEW"
                continue

            if time.monotonic() - last_frame_time > MAX_NO_FRAME_SECONDS:
                motor.stop()
                state = "WAIT_LINE"
                previous_x = None
                last_steer = 0.0
                acquire_frames = lost_frames = 0
                corner_candidate = None
                corner_frames = 0
                time.sleep(.005)
                continue


            now = time.monotonic()
            if state in ("WAIT_LINE", "FOLLOW") and has_frame:
                valid_corner = (corner_dir is not None and corner_y is not None and
                                now-last_corner_end > CORNER_COOLDOWN and
                                now-last_tracked_at < CORNER_MEMORY_SECONDS)
                if valid_corner:
                    if corner_candidate == corner_dir and now-corner_seen_at <= CORNER_CONFIRM_GAP:
                        corner_frames += 1
                    else:
                        corner_candidate = corner_dir
                        corner_frames = 1
                    corner_seen_at = now
                    corner_seen_y = corner_y
                elif now-corner_seen_at > CORNER_CONFIRM_GAP:
                    corner_candidate = None
                    corner_frames = 0
                    corner_seen_y = -1


            ready_corner = (state in ("WAIT_LINE", "FOLLOW") and has_frame and
                            corner_candidate in ("LEFT", "RIGHT") and
                            corner_frames >= CORNER_CONFIRM_FRAMES and
                            corner_seen_y >= int(mask.shape[0]*CORNER_TRIGGER_ROW) and
                            now-corner_seen_at <= CORNER_CONFIRM_GAP and
                            now-last_tracked_at <= CORNER_MEMORY_SECONDS)
            if ready_corner:
                state = "TURN_" + corner_candidate
                corner_begin = now
                corner_yaw_start = motor.yaw_snapshot()

                corner_lock = 0
                corner_early_lock = 0
                corner_had_gap = False
                corner_gap_frames = 0
                rejoin_lock = rejoin_lost = 0
                rejoin_normal_lock = 0
                rejoin_align_lock = rejoin_soft_lock = 0
                curve_recovery_lock = curve_recovery_lost = 0
                previous_x = None
                last_steer = 0.0
                print("CORNER {} y={} -> TURN (seen {:.2f}s ago)".format(
                    corner_candidate, corner_seen_y, now-last_tracked_at))
                sense = -1.0 if corner_candidate == "LEFT" else 1.0
                motor.command(CORNER_FORWARD_SPEED, steering_sign*sense*CORNER_TURN_RATE)
                corner_candidate = None
                corner_frames = 0
                continue

            if state == "WAIT_LINE":
                motor.stop()
               
                wait_normal_ok = (regular_line_is_trackable(
                    normal_found, normal_error, normal_heading)
                    if wide_state else found)
                if has_frame and wait_normal_ok:
                    acquire_frames += 1
                    if acquire_frames >= ACQUIRE_FRAMES:
                        state = "FOLLOW"
                        lost_frames = 0
                        follow_error = normal_error if wide_state else error
                        follow_heading = normal_heading if wide_state else heading
                        previous_x = normal_x if wide_state else new_x
                        print("BLACK LINE ACQUIRED -> FOLLOW")
                        forward, last_steer, curve_active = hybrid_follow_command(
                            follow_error, follow_heading, 0.0,
                            center_controller, steering_sign)
                        motor.command(forward, last_steer)
                else:
                    acquire_frames = 0
                    if has_frame:
                        previous_x = None
            elif state == "FOLLOW":
                if has_frame and found:
                    lost_frames = 0
                    forward, last_steer, curve_active = hybrid_follow_command(
                        error, heading, last_steer, center_controller, steering_sign)
                
                    approach_confirmed = (corner_candidate in ("LEFT", "RIGHT") and
                                          corner_frames >= CORNER_CONFIRM_FRAMES and
                                          corner_seen_y >= int(mask.shape[0] * CORNER_ROW_START) and
                                          corner_seen_y < int(mask.shape[0] * CORNER_TRIGGER_ROW))
                    if approach_confirmed:
                        forward = min(forward, CORNER_APPROACH_SPEED)
                    if time.monotonic() < follow_slow_until:
                        forward = min(forward, POST_CORNER_MAX_SPEED)
                    motor.command(forward, last_steer)
                else:
                    center_controller.reset()
                    curve_active = False
                    motor.stop()
                    lost_frames += 1
                    last_steer = 0.0
                    if lost_frames >= LOST_FRAMES:
                       
                        state = "RECOVER_CURVE"
                        curve_recovery_begin = time.monotonic()
                        curve_recovery_lock = 0
                        curve_recovery_lost = 0
                        previous_x = None
                        acquire_frames = 0
                        corner_candidate = None
                        corner_frames = 0
                        print("FOLLOW LINE LOST -> CURVE RECOVERY (vision-guided only)")

            elif state in ("TURN_LEFT", "TURN_RIGHT"):
                
                elapsed = time.monotonic() - corner_begin
                if has_frame:
                    corner_gap_frames = corner_gap_frames + 1 if not found else 0
                    if corner_gap_frames >= 2:
                        corner_had_gap = True

                yaw_progress = motor.yaw_since(corner_yaw_start) if corner_yaw_start else None
                enough_90_turn = (yaw_progress >= CORNER_MIN_YAW_REJOIN
                                  if yaw_progress is not None else
                                  elapsed >= CORNER_TIME_FALLBACK)
                ninety_line = (has_frame and found and
                               elapsed >= CORNER_MIN_SECONDS and enough_90_turn and
                               (corner_had_gap or elapsed >= CORNER_SAFE_REJOIN_AFTER) and
                               abs(error) <= CORNER_CENTER_TOL and
                               abs(heading) <= CORNER_HEADING_TOL)
                corner_lock = corner_lock + 1 if ninety_line else 0

                early_line = (has_frame and
                              corner_early_reacquire_allowed(
                                  elapsed, yaw_progress, corner_had_gap,
                                  normal_found, normal_error, normal_heading))
                corner_early_lock = corner_early_lock + 1 if early_line else 0

                if (corner_lock >= CORNER_LOCK_FRAMES or
                        corner_early_lock >= CORNER_EARLY_LOCK_FRAMES):
                    direction = "LEFT" if state == "TURN_LEFT" else "RIGHT"
                    early_exit = corner_early_lock >= CORNER_EARLY_LOCK_FRAMES
                    target_error = normal_error if early_exit else error
                    target_heading = normal_heading if early_exit else heading
                    target_x = normal_x if early_exit else new_x
                    state = "REJOIN_" + direction
                    rejoin_begin = time.monotonic()
                    rejoin_lock = rejoin_lost = rejoin_normal_lock = 0
                    rejoin_align_lock = rejoin_soft_lock = 0
                    previous_x = target_x
                    _, last_steer = rejoin_command(
                        target_error, target_heading, 0.0, steering_sign)
                    motor.command(CORNER_REJOIN_SPEED, last_steer)
                    print("TURN {} -> REJOIN ({}) err={:+.2f} head={:+.2f} yaw={}".format(
                        direction, "45-degree camera handoff" if early_exit else "90-degree handoff",
                        target_error, target_heading,
                        "N/A" if yaw_progress is None else "{:.2f}rad".format(yaw_progress)))
                elif (elapsed >= CORNER_MAX_SECONDS or
                      (yaw_progress is not None and yaw_progress >= CORNER_MAX_YAW)):
                    motor.stop()
                    state = "WAIT_LINE"
                    previous_x = None
                    corner_candidate = None
                    corner_frames = 0
                    last_steer = 0.0
                    acquire_frames = 0
                    last_corner_end = time.monotonic()
                    last_tracked_at = -100.0
                    last_track_anchor = None
                    print("TURN STOP {:.1f}s gyro={} -> WAIT_LINE (no confirmed new line)".format(
                        elapsed, "N/A" if yaw_progress is None else "{:.2f}rad".format(yaw_progress)))
                elif early_line:
                   
                    _, last_steer = rejoin_command(
                        normal_error, normal_heading, last_steer, steering_sign)
                    motor.command(CORNER_REJOIN_SPEED, last_steer)
                elif ninety_line:
                    _, last_steer = rejoin_command(
                        error, heading, last_steer, steering_sign)
                    motor.command(CORNER_REJOIN_SPEED, last_steer)
                elif has_frame:
                    sense = -1.0 if state == "TURN_LEFT" else 1.0
                    motor.command(CORNER_FORWARD_SPEED,
                                  steering_sign * sense * CORNER_TURN_RATE)
                else:
                    motor.stop()

            elif state in ("REJOIN_LEFT", "REJOIN_RIGHT"):
                rejoin_elapsed = time.monotonic() - rejoin_begin
                wide_valid = (has_frame and found and
                              abs(error) <= CORNER_CENTER_TOL and
                              abs(heading) <= CORNER_HEADING_TOL)
                normal_valid = (has_frame and regular_line_is_trackable(
                    normal_found, normal_error, normal_heading))
                precise, soft = rejoin_alignment_status(
                    normal_valid, normal_error, normal_heading)
                rejoin_normal_lock = rejoin_normal_lock + 1 if normal_valid else 0
                rejoin_align_lock = rejoin_align_lock + 1 if precise else 0
                rejoin_soft_lock = rejoin_soft_lock + 1 if soft else 0

                if normal_valid or wide_valid:
                    rejoin_lost = 0
                    target_error = normal_error if normal_valid else error
                    target_heading = normal_heading if normal_valid else heading
                    previous_x = normal_x if normal_valid else new_x
                    _, last_steer = rejoin_command(
                        target_error, target_heading, last_steer,
                        steering_sign, use_guide=True)
                    motor.command(CORNER_REJOIN_SPEED, last_steer)
                else:
                    rejoin_lost += 1
                    last_steer = 0.0
                    motor.stop()

               
                aligned_exit = rejoin_align_lock >= REJOIN_NORMAL_LOCK_FRAMES
                soft_exit = (rejoin_elapsed >= REJOIN_SOFT_MIN_SECONDS and
                             rejoin_soft_lock >= REJOIN_SOFT_LOCK_FRAMES)
                fallback_exit = (rejoin_elapsed >= REJOIN_FALLBACK_MIN_SECONDS and
                                 rejoin_normal_lock >= REJOIN_NORMAL_LOCK_FRAMES and
                                 abs(normal_error) <= REJOIN_FALLBACK_ERROR and
                                 abs(normal_heading) <= REJOIN_FALLBACK_HEADING)
                if normal_valid and (aligned_exit or soft_exit or fallback_exit):
                    state = "FOLLOW"
                    corner_candidate = None
                    corner_frames = 0
                    last_corner_end = time.monotonic()
                    follow_slow_until = last_corner_end + POST_CORNER_SLOW_SECONDS
                    lost_frames = 0
                    previous_x = normal_x
                    center_controller.reset()
                    forward, last_steer, curve_active = hybrid_follow_command(
                        normal_error, normal_heading, last_steer,
                        center_controller, steering_sign)
                    motor.command(min(forward, POST_CORNER_MAX_SPEED), last_steer)
                    reason = ("precise" if aligned_exit else
                              "stable" if soft_exit else "fallback")
                    print("REJOIN -> FOLLOW ({}) center_err={:+.2f} heading={:+.2f}".format(
                        reason, normal_error, normal_heading))
                elif (rejoin_lost >= CORNER_REJOIN_LOST_FRAMES or
                      rejoin_elapsed >= REJOIN_VISIBLE_MAX_SECONDS or
                      (rejoin_elapsed >= CORNER_REJOIN_MAX_SECONDS and
                       not (normal_valid or wide_valid))):
                    motor.stop()
                    state = "WAIT_LINE"
                    previous_x = None
                    last_steer = 0.0
                    acquire_frames = 0
                    corner_candidate = None
                    corner_frames = 0
                    last_corner_end = time.monotonic()
                    last_tracked_at = -100.0
                    last_track_anchor = None
                    print("REJOIN STOP: unable to align or confirm a safe line")

            elif state == "RECOVER_CURVE":
               
                recovery_elapsed = time.monotonic() - curve_recovery_begin
                normal_valid = (has_frame and regular_line_is_trackable(
                    normal_found, normal_error, normal_heading))
                wide_valid = (has_frame and found and
                              abs(error) <= CORNER_CENTER_TOL and
                              abs(heading) <= CORNER_HEADING_TOL)
                curve_recovery_lock = (curve_recovery_lock + 1 if normal_valid else 0)
                if normal_valid or wide_valid:
                    curve_recovery_lost = 0
                    target_error = normal_error if normal_valid else error
                    target_heading = normal_heading if normal_valid else heading
                    previous_x = normal_x if normal_valid else new_x
                    _, last_steer = rejoin_command(
                        target_error, target_heading, last_steer, steering_sign)
                    motor.command(CORNER_REJOIN_SPEED, last_steer)
                else:
                    curve_recovery_lost += 1
                    last_steer = 0.0
                    motor.stop()

                if curve_recovery_lock >= CURVE_RECOVERY_NORMAL_LOCK_FRAMES:
                    state = "FOLLOW"
                    follow_slow_until = time.monotonic() + POST_CORNER_SLOW_SECONDS
                    previous_x = normal_x
                    lost_frames = 0
                    center_controller.reset()
                    forward, last_steer, curve_active = hybrid_follow_command(
                        normal_error, normal_heading, last_steer,
                        center_controller, steering_sign)
                    motor.command(min(forward, POST_CORNER_MAX_SPEED), last_steer)
                    print("45-DEGREE CURVE RECOVERED -> FOLLOW err={:+.2f} head={:+.2f}".format(
                        normal_error, normal_heading))
                elif (recovery_elapsed >= CURVE_RECOVERY_MAX_SECONDS or
                      curve_recovery_lost >= CURVE_RECOVERY_MAX_NO_LINE_FRAMES):
                    motor.stop()
                    state = "WAIT_LINE"
                    previous_x = None
                    last_steer = 0.0
                    acquire_frames = 0
                    print("CURVE RECOVERY STOP: no regular line within safe recovery window")

           
            if normal_csv.active and has_frame and motor is not None:
                cmd_forward, cmd_turn = motor.command_snapshot()
                normal_csv.record(
                    state, found, error, heading, cmd_forward, cmd_turn,
                    center_controller.last_trim, curve_active, wide_state,
                    measured_vision_hz, steering_sign)
            time.sleep(0.001 if has_frame else 0.003)

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as exc:
        print("ERROR:", repr(exc))
    finally:
        if motor is not None:
            motor.close()
        normal_csv.close()
        if camera is not None:
            try:
                camera.terminate()
            except Exception as exc:
                print("Camera terminate warning:", exc)
        cv2.destroyAllWindows()
        print("PROGRAM FINISHED, verify wheels are stationary")


if __name__ == '__main__':
    main()
