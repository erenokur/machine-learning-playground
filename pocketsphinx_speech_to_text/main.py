import os
import librosa
import soundfile as sf
from pocketsphinx import *


def convert_mp3_to_txt(mp3_file):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(pocketsphinx.__file__)), 'model', 'en-us')

    # Create a decoder with the default model
    config = Decoder.default_config()
    config.set_string('-hmm', os.path.join(model_dir, 'en-us'))
    config.set_string('-lm', os.path.join(model_dir, 'en-us.lm.bin'))
    config.set_string('-dict', os.path.join(model_dir, 'cmudict-en-us.dict'))
    decoder = Decoder(config)

    try:
        # Convert MP3 to WAV with 16kHz sampling rate
        data, samplerate = sf.read(mp3_file)
        if samplerate != 16000:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
        temp_wav = "temp_audio.wav"
        sf.write(temp_wav, data, 16000, subtype="PCM_16")

        # Decode the audio
        decoder.start_utt()
        with open(temp_wav, "rb") as wav_file:
            decoder.process_raw(wav_file.read(), False, True)
        decoder.end_utt()

        os.remove(temp_wav)  # Clean up temporary file

        # Get recognized text
        hypothesis = decoder.hyp()
        text = hypothesis.hypstr if hypothesis else "No speech recognized."

        # Save text to file
        txt_file = os.path.splitext(mp3_file)[0] + ".txt"
        with open(txt_file, "w") as f:
            f.write(text)
        print(f"Converted {mp3_file} to {txt_file}")

    except Exception as e:
        print(f"Error processing {mp3_file}: {e}")


def process_mp3_files(folder_path):
    """Processes all mp3 files in the given folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            mp3_file = os.path.join(folder_path, filename)
            txt_file = os.path.splitext(mp3_file)[0] + ".txt"
            if not os.path.exists(txt_file):
                convert_mp3_to_txt(mp3_file)


# Define the audio folder path
app_path = os.getcwd()
audio_path = os.path.join(app_path, 'pocketsphinx_speech_to_text', 'audios')

if __name__ == "__main__":
    process_mp3_files(audio_path)
