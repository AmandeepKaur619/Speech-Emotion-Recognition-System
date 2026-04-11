#  Speech Emotion Recognition System

##  Project Overview

The **Speech Emotion Recognition (SER) System** is a machine learning based application that detects human emotions from speech audio signals.
The system analyzes vocal features from an audio input and predicts the emotional state of the speaker.

This project demonstrates how **audio signal processing and deep learning models** can be used to recognize emotions such as happiness, sadness, anger, and neutrality from speech.

---

##  Objectives

* Extract meaningful **audio features** from speech signals
* Train a **machine learning / deep learning model** to classify emotions
* Build an interactive interface to test emotion detection from voice input

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Librosa (Audio Processing)
* Scikit-learn
* Streamlit (Web Interface)

---

##  Project Structure

```
Speech-Emotion-Recognition-System
│
├── app.py                 # Streamlit web application
├── model.py               # Model training and preprocessing code
├── ser_model.keras        # Trained deep learning model
├── scaler.pkl             # Feature scaling object
├── label_encoder.pkl      # Emotion label encoder
├── temp.wav               # Temporary audio file
└── README.md              # Project documentation
```

---

##  Model Workflow

1. **Audio Input**
   User uploads or records a speech sample.

2. **Feature Extraction**
   Audio features such as:

   * MFCC (Mel Frequency Cepstral Coefficients)
   * Spectral features
   * Zero Crossing Rate

3. **Preprocessing**

   * Feature scaling using a trained scaler
   * Label encoding for emotion categories

4. **Emotion Prediction**
   The trained deep learning model predicts the emotion of the speaker.

---

##  How to Run the Project

### 1️ Clone the Repository

```bash
git clone https://github.com/AmandeepKaur619/Speech-Emotion-Recognition-System.git
cd Speech-Emotion-Recognition-System
```

### 2️ Create Virtual Environment

```bash
python -m venv venv
```

### 3️ Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 4️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

##  Emotions Detected

The model can classify emotions such as:

*  Happy
*  Sad
*  Angry
*  Neutral
*  Surprise

---

##  Applications

Speech emotion recognition has applications in:

* Human-computer interaction
* Mental health monitoring
* Customer service analysis
* Voice assistants
* Call center analytics

---

##  Future Improvements

* Real-time emotion detection
* Larger speech datasets
* Improved deep learning architectures
* Integration with mobile applications

---

##  Author

**Amandeep Kaur**

Aspiring Data Scientist

LinkedIn:
https://www.linkedin.com/in/amandeep-kaur-a02398315

---


