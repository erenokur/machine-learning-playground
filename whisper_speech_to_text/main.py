import whisper
import os

input_file = os.getcwd() + "/whisper_speech_to_text/input_audio/audio.mp3"
output_folder = os.getcwd() + "/whisper_speech_to_text/output_text/"

# load the model and transcribe the audio
model = whisper.load_model("large-v2")
result = model.transcribe(input_file)
# alternatively, you can give language hint to the model
# result = model.transcribe(input_file, language="Turkish")

# print the recognized text
text = result["text"]
language = result["language"]

print(f"Recognized text: {text}")
print(f"Language: {language}")

# save the recognized text to a file
with open(os.path.join(output_folder, 'recognized_text.txt'), 'w') as f:
    f.write(result["text"])