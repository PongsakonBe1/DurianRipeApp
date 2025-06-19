# main.py
import os
import threading
import platform
import numpy as np
import soundfile as sf
import librosa
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.utils import get_color_from_hex

# ใช้ tflite-runtime บน Android
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

class DurianApp(App):
    def build(self):
        self.model_path = "./best_durian_model.tflite"
        self.audio_path = "audio.wav"

        self.interpreter = None
        self.load_model()

        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        self.label = Label(text="ผลการทำนายจะแสดงที่นี่", font_size='20sp')
        layout.add_widget(self.label)

        self.record_btn = Button(text="อัดเสียง", font_size='18sp', on_press=self.record_audio)
        layout.add_widget(self.record_btn)

        self.predict_btn = Button(text="ทำนาย", font_size='18sp', on_press=self.run_inference)
        layout.add_widget(self.predict_btn)

        return layout

    def load_model(self):
        if os.path.exists(self.model_path):
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def record_audio(self, instance):
        def _record():
            try:
                from jnius import autoclass
                from android.permissions import request_permissions, Permission
                from android.storage import app_storage_path

                request_permissions([Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE])
                app_path = app_storage_path()
                self.audio_path = os.path.join(app_path, "audio.wav")

                MediaRecorder = autoclass('android.media.MediaRecorder')
                recorder = MediaRecorder()
                recorder.setAudioSource(MediaRecorder.AudioSource.MIC)
                recorder.setOutputFormat(MediaRecorder.OutputFormat.DEFAULT)
                recorder.setAudioEncoder(MediaRecorder.AudioEncoder.DEFAULT)
                recorder.setOutputFile(self.audio_path)

                recorder.prepare()
                recorder.start()

                import time
                time.sleep(5)

                recorder.stop()
                recorder.release()

                self.update_status("อัดเสียงเสร็จแล้ว")
            except Exception as e:
                self.update_status(f"อัดเสียงล้มเหลว: {e}")

        threading.Thread(target=_record).start()

    def run_inference(self, instance):
        if self.interpreter is None or not os.path.exists(self.audio_path):
            self.update_status("ไม่พบไฟล์เสียงหรือโมเดล")
            return

        try:
            y, sr = librosa.load(self.audio_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
            mfccs = np.pad(mfccs, ((0, 0), (0, 174 - mfccs.shape[1])), mode='constant')
            mfccs = mfccs[np.newaxis, ..., np.newaxis].astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], mfccs)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            confidence = float(output[1]) if len(output) == 2 else float(output[0])
            label = "สุก" if confidence >= 0.5 else "ดิบ"
            self.update_status(f"ทุเรียน: {label} ({confidence*100:.2f}%)")
        except Exception as e:
            self.update_status(f"ทำนายล้มเหลว: {e}")

    def update_status(self, msg):
        Clock.schedule_once(lambda dt: setattr(self.label, 'text', msg))

if __name__ == "__main__":
    DurianApp().run()
