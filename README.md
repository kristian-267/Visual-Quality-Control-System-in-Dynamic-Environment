# Visual-Quality-Control-System-in-Dynamic-Environment

## Overview
This repository contains a Python-based system for visual quality control in dynamic environments. It uses machine learning and image processing techniques for real-time object tracking and classification.

## Files
- `data/`: Including images used for calibrating and rectifying the cameras, data for training model, and video for tracking and predicting objects.
- `model/`: The directory where the trained model saved
- `parameters/`: The directory where the calibration and rectification parameters saved.
- `scripts/calibration.py`: Script for calibrating and rectifying the cameras.
- `scripts/classification.py`: Implements SVM algorithms to train a model.
- `scripts/model_test.py`: Used for testing the trained SVM model.
- `scripts/track_objects.py`: Implements Dense Optical Flow, Kalman Filter and SVM model to track and classify objects in a dynamic environment.

## Requirements
- Python 3.10

## Installation
Clone the repository and install dependencies listed in `requirements.txt`.

## Usage
Run the scripts in a Python environment. Ensure proper setup of the cameras for accurate results.
