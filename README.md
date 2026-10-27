# Q-Bot Autonomous Line-Following & Navigation (Milestone 2)

This repository contains the software implementation, telemetry data, and verification artifacts for Milestone 2 of our robotics project using the Quanser Q-Bot Platform (Jetson Orin Nano).

## Repository Structure
* **`src/`** — Python control scripts, thread handlers, and computer vision pipelines (`A3_Line_Following.py`).
* **`data/`** — Raw CSV telemetry logs capturing time-series error and tracking performance (`normal_line_following_...csv`).
* **`assets/`** — Performance plots (`b3_plot.png`), camera snapshots, and binary threshold masks.

## System Architecture & Control
* **Hardware:** Quanser Q-Bot Jetson Orin Nano running `QBotPlatformDriver` on `localhost:18888`.
* **Control Law:** Hybrid proportional-derivative tracking law augmented with an adaptive, bounded PID line-centering trim ($K_e = 0.24, K_h = 0.10$ for nominal tracking; $K_{p,\text{sharp}} = 0.31, K_{h,\text{sharp}} = 0.34$ for sharp turns).
* **Vision Pipeline:** OpenCV-based black-line segmentation using local adaptive row brightness thresholds and multi-row anchor detection.

## How to Run
1. Ensure the Quanser background driver is active and listening on port `18888`.
2. Run the main control script:
   ```bash
   python3 src/A3_Line_Following.py
