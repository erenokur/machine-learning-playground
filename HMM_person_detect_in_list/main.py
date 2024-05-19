import numpy as np
import librosa
from hmmlearn import hmm
import os

# Get the application startup path
app_path = os.getcwd()

# Get the paths for the training and test audio files
train_audio_path = os.path.join(app_path, 'HMM_person_detect_in_list', 'train_audio')
test_audio_path = os.path.join(app_path, 'HMM_person_detect_in_list', 'test_audio')

# Get the audio files for different persons from the train_audio folder
audio_files = {
    'person1': [os.path.join(train_audio_path, 'audio1_person1.wav'), os.path.join(train_audio_path, 'audio2_person1.wav'), os.path.join(train_audio_path, 'audio3_person1.wav')],
    'person2': [os.path.join(train_audio_path, 'audio1_person2.wav'), os.path.join(train_audio_path, 'audio2_person2.wav'), os.path.join(train_audio_path, 'audio3_person2.wav')],
    'person3': [os.path.join(train_audio_path, 'audio1_person3.wav'), os.path.join(train_audio_path, 'audio2_person3.wav'), os.path.join(train_audio_path, 'audio3_person3.wav')],
}

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpose to have time frames as rows



# Extract MFCC features for each audio file
all_sequences = []
lengths = []
for person, files in audio_files.items():
    for file in files:
        mfcc_features = extract_mfcc(file)
        all_sequences.append(mfcc_features)
        lengths.append(len(mfcc_features))

# Concatenate sequences for training
X = np.concatenate(all_sequences)

# Define the number of hidden states
n_components = 4

# Initialize the GaussianHMM model
model = hmm.GaussianHMM(n_components=n_components, n_iter=100, random_state=42)

# Fit the HMM to the data
try:
    model.fit(X, lengths)
except ValueError as e:
    print(f"Error occurred during model fitting: {e}")

# Function to evaluate if a new audio file matches the trained HMM
def evaluate_audio(file_path):
    mfcc_features = extract_mfcc(file_path)
    try:
        logprob = model.score(mfcc_features)
    except ValueError as e:
        print(f"Error occurred during evaluation: {e}")
        logprob = None
    return logprob

# Example new audio files to evaluate
test_audio = os.path.join(test_audio_path, 'test_audio1.wav')

print("Log probability of new_audio:", evaluate_audio(test_audio))