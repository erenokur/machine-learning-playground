from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import os

# download and load all models
preload_models()

output_folder = os.getcwd() + "/Bark_text_to_speech/output_audio/"

simple_text_1 = """Hello, how are you doing today?"""
simple_text_2 = """I am doing well, thank you for asking."""
simple_text_3 = """That's great to hear!"""

text_array2_person3 = generate_audio(simple_text_2, history_prompt="v2/en_speaker_3")
write_wav(os.path.join(output_folder, 'audio2_person3.wav'), SAMPLE_RATE, text_array2_person3)

text_array3_person3 = generate_audio(simple_text_3, history_prompt="v2/en_speaker_3")
write_wav(os.path.join(output_folder, 'audio3_person3.wav'), SAMPLE_RATE, text_array3_person3)

exit()

# Sample 1
text_array1_person1 = generate_audio(simple_text_1, history_prompt="v2/en_speaker_1")
write_wav(os.path.join(output_folder, 'audio1_person1.wav'), SAMPLE_RATE, text_array1_person1)

text_array2_person1 = generate_audio(simple_text_2, history_prompt="v2/en_speaker_1")
write_wav(os.path.join(output_folder, 'audio2_person1.wav'), SAMPLE_RATE, text_array2_person1)

text_array3_person1 = generate_audio(simple_text_3, history_prompt="v2/en_speaker_1")
write_wav(os.path.join(output_folder, 'audio3_person1.wav'), SAMPLE_RATE, text_array3_person1)

# Sample 2
text_array1_person2 = generate_audio(simple_text_1, history_prompt="v2/en_speaker_2")
write_wav(os.path.join(output_folder, 'audio1_person2.wav'), SAMPLE_RATE, text_array1_person2)

text_array2_person2 = generate_audio(simple_text_2, history_prompt="v2/en_speaker_2")
write_wav(os.path.join(output_folder, 'audio2_person2.wav'), SAMPLE_RATE, text_array2_person2)

text_array3_person2 = generate_audio(simple_text_3, history_prompt="v2/en_speaker_2")
write_wav(os.path.join(output_folder, 'audio3_person2.wav'), SAMPLE_RATE, text_array3_person2)

# Sample 3
text_array1_person3 = generate_audio(simple_text_1, history_prompt="v2/en_speaker_3")
write_wav(os.path.join(output_folder, 'audio1_person3.wav'), SAMPLE_RATE, text_array1_person3)

text_array2_person3 = generate_audio(simple_text_2, history_prompt="v2/en_speaker_3")
write_wav(os.path.join(output_folder, 'audio2_person3.wav'), SAMPLE_RATE, text_array2_person3)

text_array3_person3 = generate_audio(simple_text_3, history_prompt="v2/en_speaker_3")
write_wav(os.path.join(output_folder, 'audio3_person3.wav'), SAMPLE_RATE, text_array3_person3)

# Sample 4
text_array1_person4 = generate_audio(simple_text_1, history_prompt="v2/en_speaker_6")
write_wav(os.path.join(output_folder, 'audio1_person4.wav'), SAMPLE_RATE, text_array1_person4)

  