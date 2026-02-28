# Motion-Tracking-Trajectory

# Motion Trajectory Tracker (OpenCV)

## Overview

This project implements a real-time motion detection and object tracking system using classical computer vision techniques in OpenCV. The model detects moving objects in video footage, isolates a target object based on its HSV color characteristics, and tracks its motion across frames while reconstructing its full trajectory over time.

The system combines motion segmentation, color-based masking, Kalman filtering, and optical flow tracking to produce stable and continuous object tracking even under noisy visual conditions such as motion blur, temporary occlusion, or inconsistent detection.

A primary application example is tracking a tennis ball on a court, where the system marks the object’s current position while simultaneously visualizing its past movement to analyze trajectory and motion behavior.

---

## Features

* Motion detection using background subtraction (MOG2)
* HSV color-based object masking
* Real-time object localization via contour detection
* Kalman Filter prediction for smooth motion estimation
* Optical Flow (Lucas–Kanade) tracking for continuity between frames
* Trajectory visualization showing historical movement
* Real-time video processing using OpenCV

---

## System Pipeline

Each video frame passes through the following stages:

1. **Motion Detection**
   Background subtraction identifies moving regions within the frame.

2. **HSV Filtering**
   The frame is converted to HSV color space to isolate objects within a defined color range (e.g., tennis ball color).

3. **Mask Fusion**
   Motion mask and HSV mask are combined to reduce false detections.

4. **Object Detection**
   Contours are extracted to locate the moving target.

5. **State Estimation (Kalman Filter)**
   The object’s position and velocity are predicted and corrected using observed measurements.

6. **Optical Flow Tracking**
   Pixel-level tracking maintains continuity when detection briefly fails.

7. **Trajectory Reconstruction**
   Past object positions are stored and rendered as a motion trail.



## Technologies Used

* Python
* OpenCV (`cv2`)
* NumPy
* Classical Computer Vision Techniques

  * Background Subtraction
  * HSV Segmentation
  * Kalman Filtering
  * Optical Flow (PyrLK)


## Applications

* Sports analytics (ball trajectory analysis)
* Motion tracking in controlled environments
* Video surveillance research
* Object motion visualization
* Experimental computer vision pipelines

## Usage

1. Place a video file in the project directory.
2. Update the video path inside the script if necessary.
3. Run the Python script:


The program will display:

* Live tracked video output
* Binary mask showing detected object regions

## Configuration

Key parameters that can be tuned:

* HSV color range for target object
* Background subtractor sensitivity
* Minimum contour area
* Kalman process noise
* Optical flow parameters
* Trajectory length buffer

Adjusting these values allows adaptation to different lighting conditions, objects, or environments.


