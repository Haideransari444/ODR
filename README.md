Optical Digit Recognition 

Welcome to the MNIST Digit Recognition project! This project demonstrates a simple yet effective way to recognize handwritten digits using a neural network and a webcam.
Project Overview

This project utilizes the MNIST dataset, consisting of images of handwritten digits (0-9), to train a Multi-Layer Perceptron (MLP) model. The trained model can then predict digits in real-time from a webcam feed.
Features

    Digit Recognition: Recognizes digits from webcam images using a trained model.
    Real-Time Video Processing: Processes and displays video from the webcam in real-time.
    Threshold Adjustment: Allows dynamic adjustment of image preprocessing thresholds.

Technologies Used

    TensorFlow 2.2.0: For building and training the neural network model.
    OpenCV 4.4.0: For capturing and processing video from the webcam.
    NumPy: For numerical operations and image manipulation.
    Pillow: For image loading and preprocessing.

Installation

To get started with this project, ensure you have Python installed. You can then install the required packages via pip:

    Install TensorFlow, OpenCV, NumPy, and Pillow.

Usage

    Load and Train Model: If no pre-trained model is available, the script will train one using the MNIST dataset and save it.

    Run the Application: Execute the script to start the video capture application. It will display a live feed from your webcam, recognize drawn digits, and show predictions.

    Interact with the Application:
        Toggle Mode: Click on the video feed to switch between real-time video and digit recognition mode.
        Adjust Threshold: Use the provided slider to change the threshold for better digit recognition.

Code Breakdown

    Data Loading: Fetches and prepares the MNIST dataset.
    Model Training: Defines and trains the MLP model with the MNIST data.
    Prediction: Makes predictions on input images.
    Video Processing: Handles real-time video processing and digit recognition.
    Application Management: Manages model loading/training and video capture.

Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    TensorFlow: For providing the deep learning framework.
    OpenCV: For enabling real-time computer vision.
    MNIST Dataset: For providing the digit images for training.
