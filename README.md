\# AI-Based EMG Signal Classification for Muscle Activity Detection



\## Project Overview

This project uses Machine Learning to analyze 8-channel Electromyography (EMG) signals and classify muscle activities into three categories: REST, FLEX, and GRIP.



\## Objectives

\- Analyze multi-channel EMG signals.

\- Preprocess signals for classification.

\- Extract meaningful features from EMG data.

\- Classify muscle activities using a Random Forest model.

\- Visualize predictions and model performance.



\## Technologies Used

\- Python

\- Machine Learning

\- Random Forest Classifier

\- Pandas and NumPy

\- Scikit-learn

\- Matplotlib

\- Jupyter Notebook



\## Methodology



\### 1. Data Collection

An 8-channel EMG dataset containing REST, FLEX, and GRIP muscle activity recordings is used.



\### 2. Signal Preprocessing

Raw EMG signals are processed to prepare the data for feature extraction and classification.



\### 3. Window-Based Segmentation

Continuous EMG signals are divided into smaller segments to analyze muscle activity over time.



\### 4. Feature Extraction

The following features are extracted:

\- RMS (Root Mean Square)

\- MAV (Mean Absolute Value)

\- ZCR (Zero Crossing Rate)

\- WL (Waveform Length)



\### 5. Classification

A Random Forest classifier is used to identify and classify muscle activities based on the extracted features.



\## Project Structure

\- `app/` – Application and dashboard

\- `data/` – Dataset files (not included in this repository)

\- `models/` – Trained machine learning model and scaler

\- `notebooks/` – EMG signal analysis notebook

\- `src/` – Source code for model implementation



\## Applications

\- Muscle activity monitoring

\- Gesture recognition

\- Assistive technologies

\- Rehabilitation systems

\- Human-computer interaction



\## Conclusion

This project demonstrates the application of signal processing and machine learning techniques for EMG-based muscle activity classification. It provides a foundation for developing intelligent muscle activity recognition systems.

