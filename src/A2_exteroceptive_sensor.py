
from quick_start_qbot_platform import QBotPlatformCSICamera
import csv
import os
from pathlib import Path
import time

import cv2
import numpy as np

# ---------------- USER SETTINGS ----------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 400
FRAME_RATE = 40.0              # As in the tested 44_GOOD.py
TEST_DURATION = 8.0            # Seconds for each surface
SAMPLE_INTERVAL = 0.05         # Delay between polls
NO_FRAME_TIMEOUT = 3.0         # Stop if camera frames stop arriving
DARK_THRESHOLD = 100           # Grayscale threshold, 0=black, 255=white
SHOW_WINDOW = bool(os.environ.get("DISPLAY"))  # Set False if preview fails
OUTPUT_DIR = Path("downward_camera_results")

TESTS = (
    ("LIGHT", "Place a LIGHT/WHITE surface under the stationary camera."),
    ("DARK", "Place a DARK/BLACK surface under the stationary camera."),
    ("BLACK_LINE", "Place a BLACK LINE on a light background in camera view."),
)


def analyze_frame(image):
    """Measure pixel brightness and fraction of dark pixels in center ROI."""
    gray = np.asarray(image)
    if gray.ndim == 3:
        if gray.shape[2] == 1:
            gray = gray[:, :, 0]
        elif gray.shape[2] == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported camera image shape {gray.shape}")
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError(f"Invalid camera image shape {gray.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape
    # Center crop: middle 50% of image width and height
    x0, x1 = w // 4, 3 * w // 4
    y0, y1 = h // 4, 3 * h // 4
    roi = gray[y0:y1, x0:x1]

    mean_gray = float(np.mean(roi))
    dark_percent = float(np.mean(roi < DARK_THRESHOLD) * 100.0)

    # Optional black-line estimate in a 9-row horizontal stripe.
    # This is a PIXEL position, not a distance in metres.
    band_y = int(h * 0.72)
    band = gray[max(0, band_y - 4):min(h, band_y + 5), :]
    column_dark_fraction = np.mean(band < DARK_THRESHOLD, axis=0)
    dark_x = np.flatnonzero(column_dark_fraction >= 0.7)
    # Avoid reporting a center if the whole row is dark or no line is visible.
    line_x = float(np.median(dark_x)) if 6 <= dark_x.size <= w * 0.35 else None
    line_error_px = (line_x - w / 2.0) if line_x is not None else None

    return gray, (x0, y0, x1, y1), mean_gray, dark_percent, line_x, line_error_px


def run_test(camera, label, instruction, writer):
    print(f"\n--- {label} ---")
    print(instruction)
    input("Position target SAFELY, then press ENTER to record ... ")

    start = time.monotonic()
    last_frame = start
    last_print = start - 1
    samples = []
    final_gray = None
    final_view = None

    while time.monotonic() - start < TEST_DURATION:
        new_frame = bool(camera.read())
        now = time.monotonic()
        if not new_frame:
            if now - last_frame > NO_FRAME_TIMEOUT:
                raise RuntimeError(f"No new CSI frames for {NO_FRAME_TIMEOUT}s")
            time.sleep(SAMPLE_INTERVAL)
            continue

        last_frame = now
        image = np.array(camera.imageData, copy=True)
        gray, roi_xyxy, mean_gray, dark_percent, line_x, line_error = analyze_frame(image)
        elapsed = now - start
        samples.append(mean_gray)

        # These are IMAGE measurements; no physical distance is estimated.
        writer.writerow([
            label, round(elapsed, 3), round(mean_gray, 3),
            round(dark_percent, 2),
            "" if line_x is None else round(line_x, 1),
            "" if line_error is None else round(line_error, 1),
            DARK_THRESHOLD
        ])

        if now - last_print >= 0.5:
            print(f"{label:10s} t={elapsed:4.1f}s | "
                  f"brightness={mean_gray:6.1f}/255 | "
                  f"dark pixels={dark_percent:5.1f}% | "
                  f"line_x={'N/A' if line_x is None else f'{line_x:.1f}'} px")
            last_print = now

        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        x0, y0, x1, y1 = roi_xyxy
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.line(view, (gray.shape[1] // 2, 0),
                 (gray.shape[1] // 2, gray.shape[0] - 1), (255, 0, 0), 1)
        if line_x is not None and label == "BLACK_LINE":
            cv2.circle(view, (int(round(line_x)), int(gray.shape[0] * 0.72)),
                       7, (0, 0, 255), -1)
        cv2.putText(view, f"{label}  mean={mean_gray:.1f}/255  dark={dark_percent:.1f}%",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)

        final_gray = gray
        final_view = view
        if SHOW_WINDOW:
            cv2.imshow("QBot Downward-Facing Camera", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                raise KeyboardInterrupt("Camera preview closed by user")
        time.sleep(SAMPLE_INTERVAL)

    if not samples:
        raise RuntimeError(f"No valid camera data during {label}")

    # Save images for the report; overwrites old captures with same label.
    cv2.imwrite(str(OUTPUT_DIR / f"{label.lower()}_raw.png"), final_gray)
    cv2.imwrite(str(OUTPUT_DIR / f"{label.lower()}_annotated.png"), final_view)
    average = float(np.mean(samples))
    print(f"DONE {label}: {len(samples)} frames, mean brightness = {average:.2f}/255")
    print("Images saved to:", OUTPUT_DIR)
    return average


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    camera = None
    averages = {}
    try:
        print("QBot downward-facing CSI camera sensor test")
        print("MOTORS ARE NOT USED. Keep robot stationary throughout.")
        print(f"Current grayscale threshold = {DARK_THRESHOLD} (adjust if needed)")
        camera = QBotPlatformCSICamera(
            frameWidth=FRAME_WIDTH,
            frameHeight=FRAME_HEIGHT,
            frameRate=FRAME_RATE
        )

        csv_path = OUTPUT_DIR / "downward_camera_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "test", "time_s", "mean_gray_0_to_255", "dark_percent",
                "line_x_px", "line_error_px", "dark_threshold"
            ])
            for label, instruction in TESTS:
                averages[label] = run_test(camera, label, instruction, writer)
                f.flush()

        print("\n=== FINAL SUMMARY ===")
        for label, value in averages.items():
            print(f"{label:10s} : mean brightness {value:.2f} / 255")
        difference = averages["LIGHT"] - averages["DARK"]
        print(f"LIGHT - DARK difference: {difference:+.2f} gray levels")
        if difference <= 15.0:
            print("WARNING: Low/inverted contrast. Check lighting, camera target, and threshold.")
        print("CSV:", csv_path)

    except KeyboardInterrupt:
        print("\nUser stopped camera test")
    except Exception as exc:
        print("\nERROR:", exc)
    finally:
        if camera is not None:
            camera.terminate()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("Camera test finished; verify QBot is stationary.")


if __name__ == "__main__":
    main()
