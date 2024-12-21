# Pocketsphinx Speech-to-Text

## Overview

**Pocketsphinx Speech-to-Text** is a Python-based project for converting MP3 audio files into text using the Pocketsphinx speech recognition library. This tool processes audio files in a specified folder, transcribing them into text files with the same name.

## Features

- Converts MP3 audio files into text.
- Uses the CMU Sphinx (PocketSphinx) speech recognition engine.
- Processes all MP3 files in a specified folder.
- Outputs transcriptions as `.txt` files.

## Requirements

### Prerequisites

Ensure you have the following installed:

1. Python 3.8 or later
2. Required Python libraries:
   - `pocketsphinx`
   - `librosa`
   - `soundfile`
   - `numpy`

### Installation

1. Clone the repository or download the source code:

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the Pocketsphinx model files are available in the correct directory:

   - Acoustic model (`en-us`)
   - Language model (`en-us.lm.bin`)
   - Pronunciation dictionary (`cmudict-en-us.dict`)

   These files are typically included with Pocketsphinx. If missing, download them from the [CMU Sphinx website](https://cmusphinx.github.io/).

## Usage

### Folder Structure

Ensure the folder structure looks like this:

```
project_root/
  |- pocketsphinx_speech_to_text/
      |- audios/
          |- example.mp3
```

Place your MP3 files in the `audios` folder.

### Running the Program

Run the script to process all MP3 files in the `audios` folder:

```bash
python main.py
```

### Output

For each MP3 file processed, a corresponding `.txt` file with the transcription will be created in the same folder.

Example:

```
audios/
  |- example.mp3
  |- example.txt
```

## Troubleshooting

### Common Issues

1. **Incorrect or missing transcription:**

   - Ensure the MP3 file contains clear speech.
   - Verify that the audio is in English (supported by the default model).

2. **Errors during processing:**

   - Check the logs for specific error messages.
   - Verify that all required Pocketsphinx model files are correctly placed.

3. **Dependencies not found:**
   - Reinstall dependencies using `pip install -r requirements.txt`.

### Debugging

Add print statements or use logging to trace execution and identify potential issues in audio conversion or transcription.

## Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request. Please ensure your code adheres to Python best practices and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [CMU Sphinx](https://cmusphinx.github.io/) for the speech recognition engine.
- [Librosa](https://librosa.org/) for audio processing.
