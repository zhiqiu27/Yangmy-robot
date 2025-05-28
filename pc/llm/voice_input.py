import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile

# 初始化 Whisper 模型（使用 tiny 或 base 版本可加速）
model = whisper.load_model("base")

# 录音参数
DURATION = 10  # seconds
SAMPLERATE = 16000  # Hz

def record_audio(duration=DURATION, samplerate=SAMPLERATE):
    print(f"[VoiceInput] Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio

def transcribe(audio, samplerate=SAMPLERATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, samplerate, (audio * 32767).astype(np.int16))  # Save float32 as int16 PCM
        print("[VoiceInput] Transcribing...")
        result = model.transcribe(f.name, language="en")  # 自动语言识别也可设置 language=None
        return result["text"]

def listen_and_transcribe():
    audio = record_audio()
    text = transcribe(audio)
    print(f"[VoiceInput] Transcription: {text}")
    return text

if __name__ == "__main__":
    while True:
        command = input("Press Enter to record, or type 'q' to quit: ")
        if command.strip().lower() == "q":
            break
        listen_and_transcribe()
