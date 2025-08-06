import os
import tempfile
import wave

from faster_whisper import WhisperModel
from pyaudio import PyAudio, paInt16
from pynput import keyboard
from pyperclip import copy


class Phoice:
    def __init__(self):
        self.format = paInt16
        self.rate = 16000
        self.channels = 1

        self.listening = False

        self.audio = PyAudio()
        self.stream = None
        self.frames = []

        self.model = WhisperModel(
            model_size_or_path="tiny", device="cpu", compute_type="int8"
        )

    def activate(self):
        self.listening = not self.listening

        if self.listening:
            self.start()
        else:
            self.stop()

    def callback(self, data, count, time, status):
        self.frames.append(data)
        return (data, 0)

    def start(self):
        self.frames = []
        self.stream = self.audio.open(
            format=self.format,
            rate=self.rate,
            channels=self.channels,
            input=True,
            stream_callback=self.callback,
        )

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
                path = file.name

            try:
                with wave.open(path, "wb") as file:
                    file.setnchannels(self.channels)
                    file.setframerate(self.rate)
                    file.setsampwidth(self.audio.get_sample_size(self.format))
                    file.writeframes(b"".join(self.frames))

                segments, _ = self.model.transcribe(path)
                text = " ".join(segment.text.strip() for segment in segments)
                copy(text)
            finally:
                os.unlink(path)

    def listen(self):
        hotkey = keyboard.HotKey(
            {keyboard.Key.cmd_r, keyboard.Key.alt_r}, self.activate
        )

        with keyboard.Listener(
            on_press=hotkey.press, on_release=hotkey.release
        ) as listener:
            listener.join()


if __name__ == "__main__":
    phoice = Phoice()
    phoice.listen()
