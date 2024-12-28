# Google Speech-to-Text MP3 Transcription

## Overview

This Python script transcribes MP3 audio files into text using Google Speech-to-Text API. It performs the following steps:

1. Converts MP3 files to FLAC format using `pydub`.
2. Fetches the sample rate of the audio file.
3. Encodes the audio file in Base64.
4. Sends the encoded audio to Google Speech-to-Text API for transcription.
5. Saves the transcription into a `.txt` file in the same directory as the MP3 file.

## Prerequisites

### Python Version

- Python 3.8 or higher

### Required Libraries

- `requests`
- `pydub`
- `python-dotenv`

Ensure these dependencies are installed using the provided `requirements.txt` file.

### Setup

1. Install [FFmpeg](https://ffmpeg.org/), which is required by `pydub` for audio processing.
2. Create a `.env` file in the root directory and add your Google Speech-to-Text API key in the following format:
   ```
   API_KEY=your_google_api_key_here
   ```
3. Organize your project folder structure as follows:
   ```
   project_root/
   |-- google_speech_to_text/
       |-- audios/    # Folder containing MP3 audio files to transcribe
   |-- .env          # Environment file with the API key
   |-- requirements.txt
   |-- script.py     # This script
   ```

## Usage

1. Place your MP3 files in the `google_speech_to_text/audios/` directory.
2. Run the script with the command:
   ```bash
   python script.py
   ```
3. Transcriptions will be saved as `.txt` files in the same directory as the MP3 files.

## Error Handling

- If a transcription fails, the error message will be printed to the console.
- Ensure the sample rate of your audio files matches the configuration in the Google API payload.
