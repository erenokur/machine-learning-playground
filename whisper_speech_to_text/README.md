# whisper_speech_to_text

This is a simple speech to text program that uses the [Openai/wisper](https://github.com/openai/whisper). The program takes in a speech input and converts the speech to text. The program then saves the text to a .txt file.

Note: ffmpeg is required to run this program. You can download ffmpeg from [here](https://ffmpeg.org/download.html). Or you can install ffmpeg using the following commands:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

### Recommended library versions

openai-whisper-20231117

tiktoken-0.7.0

librosa==0.10.2
