import whisper
import librosa
import os
import math

input_file = os.getcwd() + "/whisper_speech_to_text/input_audio/audio.mp3"
output_folder = os.getcwd() + "/whisper_speech_to_text/output_text/"

model = whisper.load_model("base")

# load audio
audio, sample_rate = librosa.load(input_file, sr=None)
#audio = whisper.load_audio(input_file)

# calculate the number of 30-second chunks
num_chunks = math.ceil(len(audio) / (30 * sample_rate))

# initialize an empty string to store the recognized text
recognized_text = ""

for i in range(num_chunks):
    # extract the i-th 30-second chunk
    start = i * 30 * sample_rate
    end = min((i + 1) * 30 * model.sample_rate, len(audio))
    chunk = audio[start:end]

    # pad or trim the chunk to fit 30 seconds
    chunk = whisper.pad_or_trim(chunk)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)

    # decode the chunk
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # append the recognized text from the chunk
    recognized_text += result.text + " "

# print the recognized text
print(recognized_text)

# save the recognized text to a file
with open(os.path.join(output_folder, 'recognized_text.txt'), 'w') as f:
    f.write(recognized_text)