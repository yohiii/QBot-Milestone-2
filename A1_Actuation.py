from quick_start_qbot_platform import QBotPlatformDriver
import numpy as np
import time
import math

#Connect QBot
qbot = QBotPlatformDriver(
    ip="134.29.23.233", # ip qbot
    mode=1  # mode work
)

# Setup driver
distance = 0.5  # unit: m
vel = 0.1       #velocity m/s
ang_vel = 0.2      #turn left angular velocity rad/s
time_foward = distance / vel
time_spin = (math.pi / 1.65) / abs(ang_vel)


#FUNCTION
def move_forward(vel, time_vel):
    time_start = time.monotonic()
    while time.monotonic() - time_start < time_vel:
        if not qbot._handle.connected:
            raise RuntimeError("Unable to connect to QBot Driver")
        qbot.read_write_std(
            timestamp=time.time(),
            commands=np.array([vel, 0], dtype=np.float64)
        )
        time.sleep(0.01)


def move_backward(vel, time_vel):
    time_start = time.monotonic()
    while time.monotonic() - time_start < time_vel:
        if not qbot._handle.connected:
            raise RuntimeError("Unable to connect to QBot Driver")
        qbot.read_write_std(
            timestamp=time.time(),
            commands=np.array([vel*-1, 0], dtype=np.float64)
        )
        time.sleep(0.01)


def turn_left(ang_vel, time_vel):
    time_start = time.monotonic()
    while time.monotonic() - time_start < time_vel:
        if not qbot._handle.connected:
            raise RuntimeError("Unable to connect to QBot Driver")
        qbot.read_write_std(
            timestamp=time.time(),
            commands=np.array([0, ang_vel], dtype=np.float64)
        )
        time.sleep(0.01)

def turn_right(ang_vel, time_vel):
    time_start = time.monotonic()
    while time.monotonic() - time_start < time_vel:
        if not qbot._handle.connected:
            raise RuntimeError("Unable to connect to QBot Driver")
        qbot.read_write_std(
            timestamp=time.time(),
            commands=np.array([0, ang_vel*-1], dtype=np.float64)
        )
        time.sleep(0.01)


def stop_qbot():
    if qbot._handle.connected:
        for _ in range(5):
            qbot.read_write_std(
                timestamp=time.time(),
                commands=np.array([0.0, 0.0], dtype=np.float64)
            )
            time.sleep(0.02)


try:
    if not qbot._handle.connected:
        raise RuntimeError("Unable to connect to QBot Driver")

    while True:
        # Display menu
        print("\n===== QBOT MOTOR TEST =====")
        print("1: Forward 50 cm")
        print("2: Backward 50 cm")
        print("3: Turn left 90 degrees")
        print("4: Turn right 90 degrees")
        print("0: Stop and exit")

        choice = input("Choose (0-4): ").strip()

        # Exit program
        if choice == "0":
            print("Exiting motor test...")
            break

        # Check valid choice
        if choice not in ("1", "2", "3", "4"):
            print("Invalid choice. Please select 0-4.")
            continue

        try:
            if choice == "1":
                print(f"FORWARD: 50 cm, speed = {vel:.2f} m/s")
                move_forward(vel, time_foward)

            elif choice == "2":
                print(f"BACKWARD: 50 cm, speed = {-vel:.2f} m/s")
                move_backward(vel, time_foward)

            elif choice == "3":
                print(f"TURN LEFT: 90 degrees, angular speed = {ang_vel:.2f} rad/s")
                turn_left(ang_vel, time_spin)

            elif choice == "4":
                print(f"TURN RIGHT: 90 degrees, angular speed = {-ang_vel:.2f} rad/s")
                turn_right(ang_vel, time_spin)

        finally:
            stop_qbot()
            print("Movement finished. Stop command sent.")

        # Pause before returning to menu
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Ctrl+C pressed, requesting stop")

finally:
    try:
        stop_qbot()
    finally:
        qbot.terminate()

    print("Stop command sent and connection closed")