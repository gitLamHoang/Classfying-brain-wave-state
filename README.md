# Brain Wave SVM Model  

## Overview  
This project explores the use of **Electroencephalogram (EEG) data** and **Machine Learning** to classify driver states such as *drowsy* and *alert*. The system integrates a portable EEG device with Arduino hardware, collects brain wave signals, and applies Support Vector Machine (SVM) techniques for state classification.  

## Background  
- Developed under the supervision of **Professor Trinh Van Chien**, School of Information and Communication Technology, Hanoi University of Science and Technology.  
- Research conducted at a leading STEM education provider based in Hanoi, Vietnam.  
- The portable EEG device was designed, built, and connected to external devices via Arduino and C programming.  

## Data Collection  
- Collected **30+ brain wave samples** using the portable EEG device.  
- Signals transmitted and processed in real time through Arduino system.  
- Data preprocessed and analyzed in **Python**.  

## Machine Learning Approach  
- Implemented a **Support Vector Machine (SVM)** model.  
- Trained on EEG patterns to classify driver states:  
  - Drowsy
  - Alert
  - *More states can be added depending on dataset expansion*.  

## Tech Stack  
- **Hardware**: Arduino + Portable EEG device  
- **Languages**: C, Python  
- **Libraries/Frameworks**: scikit-learn, NumPy, Pandas, Matplotlib  

## Usage  
1. Connect the EEG portable device to Arduino.  
2. Stream EEG signals to the computer.  
3. Run preprocessing scripts in Python.  
4. Train and evaluate the SVM classifier.  

## Future Work  
- Expand dataset with more participants.  
- Explore deep learning models (CNN, RNN) for improved accuracy.  
- Deploy real-time driver monitoring system.  

## Acknowledgements  
Special thanks to **Professor Trinh Van Chien** for supervision and guidance throughout the project.  
