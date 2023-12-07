"""
Author: Jacob Pitsenberger
Program: main.py
Version: 1.0
Project: Face Age and Gender Detection
Date: 12/7/2023
Uses: age_gender_predictor.py

Purpose: This module serves as the entry point for running the age and gender prediction application.
         It creates an instance of the AgeGenderPredictor class from detector.py and provides options
         to process webcam feeds, image files, and video files.
    Usage:
        - Process webcam feed: python main.py --webcam
        - Process image file: python main.py --image <image_path>
        - Process video file: python main.py --video <video_path>
    Note: Ensure the necessary weights and prototxt files are available in the 'weights' directory.
"""

from age_gender_predictor import AgeGenderPredictor

if __name__ == "__main__":
    age_gender_predictor = AgeGenderPredictor()

    # Uncomment and use one of the following lines based on your choice:

    # age_gender_predictor.process_webcam_feed()  # For webcam detection

    # age_gender_predictor.process_image_file("test_files/elon_52.png")  # For image file detection
    # age_gender_predictor.process_image_file("test_files/neri_47.png")  # For image file detection

    age_gender_predictor.process_video_file("test_files/huberman_48.mp4")  # For video file detection
