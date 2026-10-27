
from quick_start_qbot_platform import QBotPlatformDriver
import numpy as np
import time
import csv

# Connect QBot
qbot = None
csv_file = None

try:
    qbot = QBotPlatformDriver(
        ip="134.29.23.233",
        mode=1
    )

    if not qbot._handle.connected:
        raise RuntimeError("QBot not connected")

    print("QBot connected!")
    print("Checking encoder data for 15 seconds...")

    start = time.monotonic()
    initial_pos = None
    last_print = 0
    fresh_count = 0

    # Create ONE CSV file to save encoder readings
    csv_file = open("encoder_test_results.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "time_s", "position_left", "position_right",
        "speed_left", "speed_right", "delta_left", "delta_right"
    ])
    csv_file.flush()

    while time.monotonic() - start < 15:

        fresh = qbot.read_write_std(
            timestamp=time.monotonic() - start,
            arm=0,
            commands=np.array([0.0, 0.0])
        )

        if fresh:
            fresh_count += 1

            pos = np.array(
                qbot.wheelPositions, dtype=float
            ).copy()

            speed = np.array(
                qbot.wheelSpeeds, dtype=float
            ).copy()

            if initial_pos is None:
                initial_pos = pos.copy()

            delta = pos - initial_pos

            # Save each new encoder sample as a separate CSV row
            writer.writerow([
                round(time.monotonic() - start, 3),
                pos[0], pos[1],
                speed[0], speed[1],
                delta[0], delta[1]
            ])
            csv_file.flush()

            if time.monotonic() - last_print >= 0.3:
                print("\nNEW DATA:", fresh)
                print("Position left & right:", pos)
                print("Speed left & right:", speed)
                print("Delta left & right:", delta)

                last_print = time.monotonic()

        time.sleep(0.02)

    print("\nFresh packets received:", fresh_count)

except KeyboardInterrupt:
    print("Test interrupted")

except Exception as e:
    print("ERROR:", e)

finally:
    if csv_file is not None:
        csv_file.close()
        print("Encoder CSV saved:", csv_file.name)

    if qbot is not None:
        try:
            if qbot._handle.connected:
                for _ in range(5):
                    qbot.read_write_std(
                        timestamp=time.monotonic(),
                        arm=0,
                        commands=np.array([0.0, 0.0])
                    )
                    time.sleep(0.02)
        finally:
            qbot.terminate()

    print("Test finished")
