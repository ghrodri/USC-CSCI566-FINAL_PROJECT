import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import torch

# Define paths and mappings
AUDIO_PATH = "/home1/ggrimald/CSCI-566-FINAL_PROJECT/Data/Audio/Audio_data"
MODEL_SAVE_PATH = "/home1/ggrimald/CSCI-566-FINAL_PROJECT/models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

emotion_mapping = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

emotion_category = {
    "Neutral": "Neutral", "Calm": "Positive", "Happy": "Positive",
    "Sad": "Negative", "Angry": "Negative", "Fearful": "Negative",
    "Disgust": "Negative", "Surprised": "Positive"
}

intensity_mapping = {"01": "Normal", "02": "Strong"}

def extract_features(file_path, max_pad_length=50000):
    try:
        y, sr = librosa.load(file_path, sr=48000)
        if len(y) == 0 or np.all(np.abs(y) < 1e-5) or np.sum(y**2) < 1e-4:
            return np.zeros(50)
        if len(y) < max_pad_length:
            y = np.pad(y, (0, max_pad_length - len(y)), 'constant')
        else:
            y = y[:max_pad_length]

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).flatten()
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, tuning=None, norm=None).flatten()
        mel = librosa.feature.melspectrogram(y=y, sr=sr).flatten()
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr).flatten()

        return np.hstack([mfccs, chroma, mel, contrast])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(50)

# Load data
X, y_emotion, y_intensity = [], [], []
for folder in os.listdir(AUDIO_PATH):
    folder_path = os.path.join(AUDIO_PATH, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code, intensity_code = parts[2], parts[3]
                if emotion_code in emotion_mapping and intensity_code in intensity_mapping:
                    file_path = os.path.join(folder_path, file)
                    features = extract_features(file_path)
                    X.append(features)
                    y_emotion.append(emotion_category[emotion_mapping[emotion_code]])
                    y_intensity.append(intensity_mapping[intensity_code])

X = np.array(X)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Encode labels
le_emotion = LabelEncoder()
le_intensity = LabelEncoder()
y_emotion_encoded = le_emotion.fit_transform(y_emotion)
y_intensity_encoded = le_intensity.fit_transform(y_intensity)

# Train-test split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_pca, y_emotion_encoded, test_size=0.2, random_state=42)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_pca, y_intensity_encoded, test_size=0.2, random_state=42)

# Model training with MLP
model_emotion = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
model_emotion.fit(X_train_e, y_train_e)

model_intensity = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
model_intensity.fit(X_train_i, y_train_i)

# Model evaluation
print("Emotion Classification Report:")
y_pred_e = model_emotion.predict(X_test_e)
print(classification_report(y_test_e, y_pred_e, target_names=le_emotion.classes_, zero_division=0))

print("Intensity Classification Report:")
y_pred_i = model_intensity.predict(X_test_i)
print(classification_report(y_test_i, y_pred_i, target_names=le_intensity.classes_, zero_division=0))

# Plot confusion matrices
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

plot_confusion_matrix(y_test_e, y_pred_e, le_emotion.classes_, "Emotion Classification Confusion Matrix", "confusion_matrix_emotion.png")
plot_confusion_matrix(y_test_i, y_pred_i, le_intensity.classes_, "Intensity Classification Confusion Matrix", "confusion_matrix_intensity.png")

# Save models using torch
torch.save(model_emotion, os.path.join(MODEL_SAVE_PATH, "Emotion_model.pth"))
torch.save(model_intensity, os.path.join(MODEL_SAVE_PATH, "Intensity_model.pth"))

print("MLP Models saved successfully.")
