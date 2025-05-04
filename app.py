import streamlit as st
import numpy as np
import librosa
import dill  # For loading models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Load the pre-trained models using dill
with open('best_svm_model.pkl', 'rb') as f:
    svm_model = dill.load(f)  # Load your best SVM model

with open('best_logreg_model.pkl', 'rb') as f:
    logreg_model = dill.load(f)  # Load your best Logistic Regression model

with open('best_mlp_model.pkl', 'rb') as f:
    perceptron_model = dill.load(f)  # Load your best Perceptron model

with open('best_dnn_model.pkl', 'rb') as f:
    dnn_model = dill.load(f)  # Load your best DNN model (MLPClassifier)


# Function to extract MFCC features from an audio file
def extract_features_with_folder_labels(file, n_mfcc=40, max_len=862):
    # Load audio file
    audio, sr = librosa.load(file, sr=None)

    # Extract MFCC features from the audio signal
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # If the MFCC matrix is smaller than max_len, pad it
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        # If it's larger, truncate it
        mfcc = mfcc[:, :max_len]

    # Flatten the MFCC matrix to 1D (this should match the training format)
    return mfcc.flatten()


# Function to predict the label for audio files
def predict_audio(model, file):
    # Extract features from the uploaded audio file
    mfcc_features = extract_features_with_folder_labels(file)

    # Ensure the feature size is correct (matching what was used in training)
    print(f"Extracted features size: {mfcc_features.shape}")

    # Predict using the selected model
    prediction = model.predict([mfcc_features])

    return 'Bonafide' if prediction == 0 else 'Deepfake'


# Streamlit UI
st.title('Deepfake Detection & Multi-Label Defect Prediction')
st.write("Choose a model and upload an audio file for deepfake detection or input features for defect prediction.")

# Model Selection for Deepfake Detection
model_choice = st.selectbox("Choose a model for Deepfake Detection",
                            ("SVM", "Logistic Regression", "Perceptron", "DNN"))

# Upload Audio File for Deepfake Prediction
audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

if audio_file is not None:
    # Predict the audio file label (Bonafide or Deepfake) using the chosen model
    if model_choice == "SVM":
        prediction = predict_audio(svm_model, audio_file)
    elif model_choice == "Logistic Regression":
        prediction = predict_audio(logreg_model, audio_file)
    elif model_choice == "Perceptron":
        prediction = predict_audio(perceptron_model, audio_file)
    elif model_choice == "DNN":
        prediction = predict_audio(dnn_model, audio_file)

    # Display the prediction result
    st.write(f"The uploaded audio file is predicted as: {prediction}")

# Multi-Label Defect Prediction
st.subheader("Software Defect Prediction")

# Input feature vector for defect prediction
feature_input = st.text_input("Enter the feature vector (comma-separated)")

if feature_input:
    features = np.array([float(x) for x in feature_input.split(',')])  # Convert input to a numeric array

    # Predict using the selected model
    defect_model_choice = st.selectbox("Choose a model for defect prediction",
                                       ("SVM", "Logistic Regression", "Perceptron", "DNN"))

    if defect_model_choice == "SVM":
        defect_prediction = predict_defects(svm_model, features)
    elif defect_model_choice == "Logistic Regression":
        defect_prediction = predict_defects(logreg_model, features)
    elif defect_model_choice == "Perceptron":
        defect_prediction = predict_defects(perceptron_model, features)
    elif defect_model_choice == "DNN":
        defect_prediction = predict_defects(dnn_model, features)

    st.write(f"Predicted defects: {defect_prediction}")
