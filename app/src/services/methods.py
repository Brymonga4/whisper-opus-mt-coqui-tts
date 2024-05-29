import os
import numpy as np
import pandas as pd
import whisper
from TTS.api import TTS
from transformers import pipeline
from services.util import Utilities

class Methods:
    @staticmethod
    def options_translate_transcribe(language="english", task="translate" ):
        options = whisper.DecodingOptions(language=language, task=task, without_timestamps=True)
        return options
    
    #['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
    @staticmethod
    def generate_audio(device, text, speaker_wav_path, language, output_file_path):
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
        # Generar el archivo de audio a partir del texto
        tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language=language, file_path=output_file_path)

    @staticmethod
    def process_audio(model, audio_data_mic_path, audio_data_file_path, lang="en"):

        if audio_data_mic_path is None or os.path.getsize(audio_data_mic_path) == 0:
            audio_data = audio_data_file_path
        else:
            audio_data = audio_data_mic_path

        options = whisper.DecodingOptions(language="english", task="translate", without_timestamps=True)

        audio = whisper.load_audio(audio_data)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        result = model.decode(mel, options)

        print(result.text)

        output_file_path = "output.wav"

        Methods.generate_audio(result.text, audio_data, "en", output_file_path)

        return result.text, output_file_path
    
    @staticmethod
    def process_audio_locally(device, model, audio_sample_path, output_file_path, lang="en"):

        # Idioma al que traducir
        options = whisper.DecodingOptions(language="french", task="translate", without_timestamps=True)

        audio = whisper.load_audio(audio_sample_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        result = model.decode(mel, options)

        print(result.text)

        # Paso para traducir 

        Methods.generate_audio(device, result.text, audio_sample_path, "es", output_file_path)

    @staticmethod
    def transcribe_audio_to_text(model, audio_sample_path):
        # Idioma al que traducir
        options = whisper.DecodingOptions(language="english", task="transcribe", without_timestamps=True)

        audio = whisper.load_audio(audio_sample_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        result = model.decode(mel, options)
        print(result.text)

    @staticmethod
    def process_audio_locally_Marian(device, model, audio_sample_path, output_file_path, lang_orig, lang_target="en"):

        # Cargar el audio en Whisper para iniciar la trasncripción
        audio = whisper.load_audio(audio_sample_path)
        audio = whisper.pad_or_trim(audio)

        # Procesar el audio para que el modelo lo entienda mejor  
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detectar el idioma hablado
        _, probs = model.detect_language(mel)
        print(model.detect_language(mel))
        print(f"Detected language: {max(probs, key=probs.get)}")
        detected_language = max(probs, key=probs.get)

        # Transcripción del audio

        options = whisper.DecodingOptions(language=detected_language, task="transcribe", without_timestamps=True)
        result = model.decode(mel, options)
        transcription = result.text

        # Resultado de la transcripción
    
        print(transcription)

        # Paso para traducir

        translation = Methods.translation_from_esp_to_lang(lang_target, transcription)

        # Resultado de la traducción
        print(translation)
        if len(translation) == 0: 
            translation = "Idioma no sorpotado"
            lang_target = "es"

        Methods.generate_audio(device, translation, audio_sample_path, lang_target, output_file_path)

    @staticmethod
    def translation_from_esp_to_lang(lang, transcription):
        if Utilities.supported_lang(lang):
            try:
                translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-es-{lang}")
                result = translator(transcription) 
                return result[0]['translation_text']
            except Exception as e:
                print("Error durante la traducción :", e)
        else:
            return ""