# Age and Gender Prediction

## Overview

This Python program utilizes computer vision techniques to predict the age and gender of individuals in images or video streams. It employs pre-trained models for face detection, gender prediction, and age prediction.

## Example Images

### Elon Musk
Age and gender detection on an image of Elon Musk.
![Elon Musk](output_files/elon_52.png_preds.jpg)

### Neri Oxman
Age and gender detection on an image of Neri Oxman 
![Huberman Video](output_files/neri_47.png_preds.jpg)

## Prerequisites

Make sure you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

Ensure the required model files are available in the weights directory. Refer to the documentation or model sources for downloading these files.

## Usage

If you want to run the program directly, you can use the following lines in the `main.py` file:

```python
if __name__ == "__main__":
    age_gender_predictor = AgeGenderPredictor()

    # Uncomment and use one of the following lines based on your choice:

    # age_gender_predictor.process_webcam_feed()  # For webcam detection

    # age_gender_predictor.process_image_file("test_files/elon_52.png")  # For image file detection

    age_gender_predictor.process_video_file("test_files/huberman_48.mp4")  # For video file detection
```

Make sure to uncomment and modify the lines according to your specific use case. You can choose between webcam detection, image file detection, or video file detection by uncommenting the respective line. Adjust the file paths as needed.


## Configuration
Adjust the configuration and settings in the main.py file according to your requirements. You may need to modify paths, confidence thresholds, or other parameters.

## Acknowledgments
This program uses pre-trained models for face detection, gender prediction, and age prediction. Refer to the model sources for details.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.