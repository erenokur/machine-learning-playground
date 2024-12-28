import requests
from pydub import AudioSegment
from pydub.utils import mediainfo
import base64
import os
from dotenv import load_dotenv

def mp3_to_flac(mp3_file, flac_file):
    """Convert an MP3 file to FLAC format."""
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(flac_file, format="flac")

def get_audio_sample_rate(file_path):
    """Get the sample rate of an audio file."""
    info = mediainfo(file_path)
    return int(info['sample_rate'])

def transcribe_audio_with_api_key(mp3_file, api_key):
    """Transcribe an MP3 file using Google Speech-to-Text API with API key."""
    # Convert MP3 to FLAC
    flac_file = mp3_file.replace(".mp3", ".flac")
    mp3_to_flac(mp3_file, flac_file)

    sample_rate = get_audio_sample_rate(flac_file)

    # Read the FLAC file and encode it in base64
    with open(flac_file, "rb") as f:
        audio_content = base64.b64encode(f.read()).decode("utf-8")

    # Speech-to-Text API URL
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"

    # Prepare the API request payload
    payload = {
        "config": {
            "encoding": "FLAC",
            "sampleRateHertz": sample_rate,  # Ensure this matches your MP3 file's sample rate
            "languageCode": "en-US"   # Change this to your preferred language
        },
        "audio": {
            "content": audio_content
        }
    }

    # Make the API request
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        results = response.json().get("results", [])
        transcription = " ".join(result["alternatives"][0]["transcript"] for result in results)
        return transcription
    else:
        return f"Error: {response.status_code}, {response.text}"


def process_mp3_files(folder_path, api_key_path):
    """Processes all mp3 files in the given folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            mp3_file = os.path.join(folder_path, filename)
            txt_file = os.path.splitext(mp3_file)[0] + ".txt"
            if not os.path.exists(txt_file):
                text = transcribe_audio_with_api_key(mp3_file, api_key_path)
                # Save text to file
                txt_file = os.path.splitext(mp3_file)[0] + ".txt"
                with open(txt_file, "w") as f:
                    f.write(text)
                print(f"Converted {mp3_file} to {txt_file}")

# Define the audio folder path
app_path = os.getcwd()
audio_path = os.path.join(app_path, 'google_speech_to_text', 'audios')
load_dotenv()

if __name__ == "__main__":
    api_key = os.getenv("API_KEY")
    try:
        process_mp3_files(audio_path, api_key)
        print("Process Completed")
    except Exception as e:
        print("An error occurred:", e)
