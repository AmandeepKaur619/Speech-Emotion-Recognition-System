import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import joblib

DATA_PATH = r"C:\Users\HP\Downloads\archive"

audio_files = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            audio_files.append(file_path)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

emotion_labels = []

for file in audio_files:
    fname = os.path.basename(file)
    parts = fname.split("-")
    emotion_code = parts[2]
    emotion_labels.append(EMOTION_MAP.get(emotion_code))

df = pd.DataFrame({"filepath": audio_files, "emotion": emotion_labels})

durations = []

for fp in df["filepath"][:300]:
    audio, sr = librosa.load(fp, sr=None)
    durations.append(librosa.get_duration(y=audio, sr=sr))

file_example = df["filepath"][0]
sig, sr = librosa.load(file_example)

spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(sig)), ref=np.max)

mfccs = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=20)
emotion_mfcc = {}

for emotion in df["emotion"].unique():
    mfcc_list = []
    subset = df[df["emotion"] == emotion]["filepath"][:10]
    
    for fp in subset:
        y, sr = librosa.load(fp)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_list.append(np.mean(mfcc, axis=1))
        
    emotion_mfcc[emotion] = np.mean(np.vstack(mfcc_list), axis=0)

import librosa
import librosa.display

file_path = df["filepath"][4]     
emotion = df["emotion"][4]

y, sr = librosa.load(file_path, sr=None)

S = librosa.stft(y)
S_db = librosa.amplitude_to_db(abs(S))

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            
        return mfcc.flatten()
    except Exception as e:
        print("Error:", file_path, "\n", e)
        return None
    
import soundfile as sf

file = r"C:/Users/HP/Downloads/archive/audio_speech_actors_01-24/Actor_24/03-01-08-02-02-02-24.wav"

signal, sr = librosa.load(file, sr=None)
print("Loaded:", len(signal))

features = []
labels = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            parts = file.split("-")
            emotion_code = parts[2]
            emotion = EMOTION_MAP.get(emotion_code)
            
            feat = extract_features(file_path)
            if feat is not None:
                features.append(feat)
                labels.append(emotion)
print("Total Samples:", len(features))

X = np.array(features)
y = np.array(labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", test_acc)

model.save("ser_model.keras")
joblib.dump(label_encoder, "label_encoder.pkl")

joblib.dump(scaler, "scaler.pkl")

def predict_emotion(file):
    features = extract_features(file)
    features = np.expand_dims(features, axis=0)

    pred = model.predict(features)
    class_index = np.argmax(pred)
    emotion = label_encoder.inverse_transform([class_index])[0]

    return emotion





