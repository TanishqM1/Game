# Game Detection Project

This project utilizes Python libraries like OpenCV to collect and process visual data, and Python with TensorFlow to collect audio data, train machine learning models, and evaluate audio data.

## Goal:
The primary objective of this project is to use machine learning models to evaluate both audio and visual data, identifying significant game events such as Molotovs, grenades, Zeus strikes, and health changes. These events are then communicated to an ESP32 controller, to send signals based on detected events.

## Technologies Used:
- **TensorFlow w/ Keras**: Used for building and training machine learning models to classify audio cues.
- **Python**: The main language used for scripting, processing data, and model development.
- **cv2 (OpenCV)**: Used to capture and process visual data, identifying important in-game visual cues like health changes and blood.
- **Numpy**: Used for handling and processing the audio data, particularly for feature extraction and manipulation.
- **ESP32**: Used to send signals based on detected events, enabling interaction with hardware.

## Features:
- **Audio Detection**: 
  - Detects key audio signals such as Molotovs, grenades, and Zeus strikes using trained machine learning models.
  - Processes audio using a custom pipeline to convert raw audio data into spectrograms for model input.

- **Visual Detection**:
  - Identifies visual cues such as blood splatters and health changes using computer vision techniques (OpenCV).
  - Captures live video from the game environment for real-time analysis.

- **ESP32 Communication**: 
  - Sends signals based on the results of the audio and visual detection models to an ESP32 microcontroller.

## Development Goal:
- Learn and apply video and audio processing techniques.
- Collect and clean game-related data for machine learning model training.
- Create Python scripts for data processing and model evaluation.
- Gain experience working with ESP32 microcontrollers to integrate machine learning results into real-world applications.


## Developers
- Tanishq Mehta
- Randiv Adhihetty
- Kehan Hettiarachchi 

## Credits
- Thank you to [Nicholas Renotte](https://www.youtube.com/watch?v=ZLIPkmmDJAc) for the audio machine learning tutorial

