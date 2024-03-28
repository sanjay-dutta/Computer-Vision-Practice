# Pedestrian Detection with 4 Different Computer Vision Techniques

## Overview

This project explores pedestrian detection using four different computer vision techniques. Each method leverages unique algorithms and models to detect pedestrians in images or video streams. Understanding and comparing these methods can provide insights into the strengths and weaknesses of each approach under various conditions.

## Methods Overview

### Method 1: Background Subtraction + Contour Extraction

This method involves distinguishing moving pedestrians from the static background by subtracting the current frame from a background model and then extracting contours from the resulting difference image.

### Method 2: Haar Cascades (Viola-Jones Classifiers)

Utilizes pre-trained Haar Cascade classifiers designed for detecting pedestrians. This method is effective for real-time detection due to its fast processing time.

### Method 3: Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM)

This technique uses HOG descriptors to capture edge or gradient structure that is characteristic of human forms and an SVM to classify these descriptors as pedestrian or non-pedestrian.

### Method 4: Single Shot Detector (SSD) with MobileNet

Employs a pre-trained SSD model with MobileNet as the backbone for feature extraction. This deep learning approach is capable of detecting pedestrians in various poses and lighting conditions.


