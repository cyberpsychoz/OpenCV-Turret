# OpenCV-Based Turret
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)](https://github.com/cyberpsychoz/OpenCV-Turret)


> **Version 0.4** — *Added safety control and best target-tracking.*

This is an experimental project for a test turret that detects people wearing bandanas in video streams and can be used as a basis for target identification systems. The code is written in Python using OpenCV.

## Features
- **YOLOv8 via Ultralytics**: Accurate person detection
- **MediaPipe**: Body keypoint detection for precise head and hand localization
- **Color classification**: Detection of bandanas across the entire body
- **Kalman Filter**: Target motion prediction
- **GPU acceleration**: CUDA support (NVIDIA only)
- **Progress bar**: Shows processing percentage

## Warning
- **Rough implementation**
  Many features need refinement and optimization

## Installation

```bash
# Install dependencies
pip install opencv-python numpy mediapipe ultralytics filterpy rich

# For Windows (headless version, no GUI)
pip install opencv-python-headless numpy mediapipe ultralytics filterpy rich
```

## Usage

1. Place test videos into the `test_videos` folder
2. Run:
   ```bash
   python main.py
   ```
3. Processed videos will be saved to the `output` folder

## Example Output
```
[INFO] Found 2 videos for processing:
1. test1.mp4
2. test2.mp4

[INFO] Processing: test1.mp4
Format: 1280x720, 30 FPS, 900 frames
[##################################################--------------------------] 67.2% | Elapsed: 00:00:18 | Remaining: 00:00:09
[INFO] Done: processed_test1.avi | Time taken: 00:00:27
```

## Roadmap
- [~] Integrate various modules to improve target detection
- [ ] Connect to a real turret via Arduino/Raspberry Pi
- [~] Improve color classification algorithm
- [ ] Add multi-target tracking
- [ ] Create a graphical user interface (GUI)
- [ ] Support RTSP camera streams
- [ ] Test on mobile devices

## Contributing
All improvements are welcome! To contribute:
1. Fork the project
2. Create a new branch (`git checkout -b feature/god-help`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
MIT License
This project was created for educational purposes only. Do not use it in combat systems.

## Contact
- Telegram: @terminisle
- Discord: terminisle

### Note for Windows users with AMD GPUs
OpenCV does not support CUDA on AMD GPUs. The code will automatically switch to CPU mode. For best performance, use a Windows system with an NVIDIA GPU or switch to Linux.
