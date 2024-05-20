# Bark_text_to_speech

This is a simple text to speech program that uses the [suno-ai/bark](https://github.com/suno-ai/bark) API to convert text to speech. The program takes in a text input and converts the text to speech. The program then saves the speech to a .waw file.

All available text to speech models can be found in the https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c

Currently, English, Chinese, German, Hindi, Italian Japanese, Korean, Polish, Portuguese, Russian, Spanish, Turkish and French are supported.

Bark is a GPT-style model. As such, it may take some creative liberties in its generations, resulting in higher-variance model outputs than traditional text-to-speech approaches.

### Pre-Trained Models

Pre-trained model in machine learning is a model that has already been trained on a large dataset and saved for later use. Instead of building and training a model from scratch for your specific task, you can use a pre-trained model and fine-tune it to suit your needs. This approach leverages the knowledge the model has already acquired during its initial training, which often involves substantial computational resources and time.

### Text-Prompted Generative Audio Model

A Text-Prompted Generative Audio Model is a type of machine learning model designed to generate audio content based on textual input. These models are typically used for tasks such as text-to-speech (TTS), music generation, and sound effect creation. Bark is a transformer-based text-to-audio model that can generate high-quality audio from text prompts. Bark uses [hugging face's transformers library](https://github.com/huggingface/transformers) to generate audio from text prompts.

### What is a transformer in machine learning?

A transformer is a type of deep learning model that is used for tasks such as natural language processing (NLP) and computer vision. Transformers are based on a self-attention mechanism that allows the model to weigh the importance of different input tokens when making predictions. This mechanism enables transformers to capture long-range dependencies in the data and generate high-quality outputs. Transformers have been widely adopted in the field of machine learning due to their ability to achieve state-of-the-art performance on a wide range of tasks.

### Recommended library versions

bark==0.0.1

scipy==1.10.1

ipython==8.12.0
