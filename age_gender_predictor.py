"""
Author: Jacob Pitsenberger
Program: main.py
Version: 1.0
Project: Face Age and Gender Detection
Date: 12/7/2023
Uses: age_gender_predictor.py

Purpose: This module defines the AgeGenderPredictor class, which encapsulates the functionality for predicting
         age and gender from webcam feeds, image files, and video files. It utilizes pre-trained models for
         face detection, gender prediction, and age prediction.

         The class includes methods for processing webcam feeds, image files, and video files, as well as
         helper functions for face detection and prediction.

    Note: Make sure to have the required model files in the 'weights' directory before using this module.
"""

from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import os



class AgeGenderPredictor:
    """
    Age and gender prediction class using OpenCV and pre-trained models.
    """
    DRAWING_COLOR = (0, 0, 255)
    FONT_SCALE = 0.4
    FONT_THICKNESS = 1
    def __init__(self):
        """
        Initialize the AgeGenderPredictor with pre-trained models.
        """
        # Define gender and age prediction model mean values
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        # Define gender labels
        self.GENDER_LIST = ['Male', 'Female']
        # Define age labels
        self.AGE_INTERVALS = ['(0, self.FONT_THICKNESS)', '(4, 6)', '(8, 12)', '(15, 20)',
                              '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

        # Paths for the face detection model
        self.FACE_PROTO = "weights/deploy.prototxt.txt"
        self.FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        # Load the face detection model
        self.face_net = cv2.dnn.readNetFromCaffe(self.FACE_PROTO, self.FACE_MODEL)

        # Paths for gender and age prediction models
        self.GENDER_MODEL = 'weights/deploy_gender.prototxt'
        self.GENDER_PROTO = 'weights/gender_net.caffemodel'
        self.AGE_MODEL = 'weights/deploy_age.prototxt'
        self.AGE_PROTO = 'weights/age_net.caffemodel'

        # Load the gender and age prediction models
        self.gender_net = cv2.dnn.readNetFromCaffe(self.GENDER_MODEL, self.GENDER_PROTO)
        self.age_net = cv2.dnn.readNetFromCaffe(self.AGE_MODEL, self.AGE_PROTO)

    def get_faces(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frame.

        Parameters:
        - frame (np.ndarray): Input frame for face detection.
        - confidence_threshold (float): Confidence threshold for face detection.

        Returns:
        - List[Tuple[int, int, int, int]]: List of faces represented as (start_x, start_y, end_x, end_y) tuples.
        """
        # convert the frame into a blob to be ready for NN input
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
        # set the image as input to the NN
        self.face_net.setInput(blob)
        # perform inference and get predictions
        output = np.squeeze(self.face_net.forward())
        # initialize the result list
        faces = []
        # Loop over the faces detected
        for i in range(output.shape[0]):
            confidence = output[i, 2]
            if confidence > confidence_threshold:
                box = output[i, 3:7] * \
                      np.array([frame.shape[1], frame.shape[0],
                                frame.shape[1], frame.shape[0]])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(int)
                # widen the box a little
                start_x, start_y, end_x, end_y = start_x - \
                                                 10, start_y - 10, end_x + 10, end_y + 10
                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = 0 if end_x < 0 else end_x
                end_y = 0 if end_y < 0 else end_y
                # append to our list
                faces.append((start_x, start_y, end_x, end_y))
        return faces

    def get_gender_predictions(self, face_img: np.ndarray) -> np.ndarray:
        """
        Get gender predictions for the given face image.

        Parameters:
        - face_img (np.ndarray): Input face image.

        Returns:
        - np.ndarray: Gender predictions.
        """
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=self.MODEL_MEAN_VALUES, swapRB=False, crop=False
        )
        self.gender_net.setInput(blob)
        return self.gender_net.forward()

    def get_age_predictions(self, face_img: np.ndarray) -> np.ndarray:
        """
        Get age predictions for the given face image.

        Parameters:
        - face_img (np.ndarray): Input face image.

        Returns:
        - np.ndarray: Age predictions.
        """
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=self.MODEL_MEAN_VALUES, swapRB=False
        )
        self.age_net.setInput(blob)
        return self.age_net.forward()

    def process_webcam_feed(self, webcam_index: int = 0) -> None:
        """
        Process the webcam feed for age and gender detection.

        Parameters:
        - webcam_index (int): Index of the webcam to use.
        """
        cap = cv2.VideoCapture(webcam_index)
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Get faces from the frame
                faces = self.get_faces(frame)

                for face in faces:
                    # Extract face region
                    start_x, start_y, end_x, end_y = face
                    face_img = frame[start_y:end_y, start_x:end_x]

                    # Predict age and gender
                    age_preds = self.get_age_predictions(face_img)
                    gender_preds = self.get_gender_predictions(face_img)

                    # Find the indices with the highest prediction scores
                    i = gender_preds[0].argmax()
                    gender = self.GENDER_LIST[i]  # 'Male' or 'Female'
                    gender_confidence_score = gender_preds[0][i]  # Confidence score (e.g., 0.85)

                    i = age_preds[0].argmax()
                    age = self.AGE_INTERVALS[i]  # e.g., '(25, 32)'
                    age_confidence_score = age_preds[0][i]  # Confidence score (e.g., 0.75)

                    # Display the result on the frame
                    cv2.putText(frame, f"Age: {age} ({age_confidence_score:.2f})", (start_x, start_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.DRAWING_COLOR, self.FONT_THICKNESS)
                    cv2.putText(frame, f"Gender: {gender} ({gender_confidence_score:.2f})", (start_x, end_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.DRAWING_COLOR, self.FONT_THICKNESS)

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), self.DRAWING_COLOR, self.FONT_THICKNESS)

                # Display the resulting frame
                cv2.imshow('Webcam - Age and Gender Detection', frame)

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as webcam_capture_error:
            print(f"Error capturing from webcam: {webcam_capture_error}")

        finally:
            # When everything is done, release the capture
            cap.release()
            cv2.destroyAllWindows()

    def process_image_file(self, image_path: str) -> None:
        """
        Process a single image file for age and gender detection.

        Parameters:
        - image_path (str): Path to the input image file.
        """
        try:
            # Load image
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Error: Couldn't read the image from {image_path}.")
                return

            # Get faces from the frame
            faces = self.get_faces(frame)

            for face in faces:
                try:
                    # Extract face region
                    start_x, start_y, end_x, end_y = face
                    face_img = frame[start_y:end_y, start_x:end_x]

                    # Predict age and gender
                    age_preds = self.get_age_predictions(face_img)
                    gender_preds = self.get_gender_predictions(face_img)

                    # Find the indices with the highest prediction scores
                    i = gender_preds[0].argmax()
                    gender = self.GENDER_LIST[i]  # 'Male' or 'Female'
                    gender_confidence_score = gender_preds[0][i]  # Confidence score (e.g., 0.85)

                    i = age_preds[0].argmax()
                    age = self.AGE_INTERVALS[i]  # e.g., '(25, 32)'
                    age_confidence_score = age_preds[0][i]  # Confidence score (e.g., 0.75)

                    # Display the result on the frame
                    cv2.putText(frame, f"Age: {age} ({age_confidence_score:.2f})", (start_x, start_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.DRAWING_COLOR, self.FONT_THICKNESS)
                    cv2.putText(frame, f"Gender: {gender} ({gender_confidence_score:.2f})", (start_x, end_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.DRAWING_COLOR, self.FONT_THICKNESS)

                    # Draw rectangle around the face
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), self.DRAWING_COLOR, self.FONT_THICKNESS)

                except Exception as face_processing_error:
                    print(f"Error processing face: {face_processing_error}")

            # Display the resulting frame
            cv2.imshow('Image - Age and Gender Detection', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save output image with predictions
            output_filename = os.path.join('output_files', os.path.basename(image_path) + '_preds.jpg')
            cv2.imwrite(output_filename, frame)
            print(f"Output saved to: {output_filename}")

        except Exception as image_processing_error:
            print(f"Error processing image file: {image_processing_error}")

    def get_video_detections(self, faces: List[Tuple[int, int, int, int]], frame: np.ndarray) -> np.ndarray:
        """
        Get video frame with age and gender detections drawn on it.

        Parameters:
        - faces (List[Tuple[int, int, int, int]]): List of faces.
        - frame (np.ndarray): Input video frame.

        Returns:
        - np.ndarray: Output video frame with detections.
        """
        try:
            for face in faces:
                # Extract face region
                start_x, start_y, end_x, end_y = face
                face_img = frame[start_y:end_y, start_x:end_x]

                # Predict age and gender
                age_preds = self.get_age_predictions(face_img)
                gender_preds = self.get_gender_predictions(face_img)

                # Find the indices with the highest prediction scores
                i = gender_preds[0].argmax()
                gender = self.GENDER_LIST[i]  # 'Male' or 'Female'
                gender_confidence_score = gender_preds[0][i]  # Confidence score (e.g., 0.85)

                i = age_preds[0].argmax()
                age = self.AGE_INTERVALS[i]  # e.g., '(25, 32)'
                age_confidence_score = age_preds[0][i]  # Confidence score (e.g., 0.75)

                # Display the result on the frame
                cv2.putText(frame, f"Age: {age} ({age_confidence_score:.2f})", (start_x, start_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.DRAWING_COLOR, self.FONT_THICKNESS)
                cv2.putText(frame, f"Gender: {gender} ({gender_confidence_score:.2f})", (start_x, end_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.DRAWING_COLOR, self.FONT_THICKNESS)

                # Draw rectangle around the face
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), self.DRAWING_COLOR, self.FONT_THICKNESS)
                # Return the frame with detections drawn on it.
            return frame

        except Exception as e:
            print(f"Error getting video file detection: {e}")
            raise

    def process_video_file(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)

        try:
            # Get the dimensions of the video frames.
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read the first frame from the video.")
                return

            H, W, _ = frame.shape

            # Specify the path to save the video with found detections.
            base_filename = Path(video_path).stem
            output_filename = os.path.join("output_files", f"{base_filename}_preds.mp4")

            # Initialize our video writer for saving the output video.
            out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)),
                                  (W, H))

            while ret:
                # Get faces from the frame
                faces = self.get_faces(frame)

                out.write(frame)

                # Keep reading the frames from the video file until they have all been processed.
                ret, frame = cap.read()

                # Handle the case when frame is None
                if frame is None:
                    break

                # Draw detections on the video frames
                frame = self.get_video_detections(faces, frame)

                # Display the resulting frame
                cv2.imshow('Video - Age and Gender Detection', frame)

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as video_capture_error:
            print(f"Error capturing from video file: {video_capture_error}")

        finally:
            # Release the video capture and writer objects
            cap.release()
            out.release()
            cv2.destroyAllWindows()
