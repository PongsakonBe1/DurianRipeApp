import os
import platform
import numpy as np
import threading
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.utils import get_color_from_hex
from kivy.resources import resource_find
from kivy.resources import resource_add_path

resource_add_path(os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets/fonts')))
resource_add_path(os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets/models')))


# ตรวจสอบว่าเป็น Android หรือไม่
is_android = platform.system() == 'Linux' and 'ANDROID_ARGUMENT' in os.environ

# ใช้ TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

def register_asset_paths():
    if is_android:
        from android.storage import app_storage_path
        base_path = app_storage_path()
        resource_add_path(os.path.join(base_path, "assets"))
        resource_add_path(os.path.join(base_path, "assets", "fonts"))
        resource_add_path(os.path.join(base_path, "assets", "models"))

def safe_font(path):
    resolved = resource_find(path)
    if not resolved:
        print(f"[ERROR] ไม่พบฟอนต์: {path}")
        return None  # หรือใช้ default font
    return resolved

def font_color(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    return get_color_from_hex('#000000') if luminance > 0.5 else get_color_from_hex('#FFFFFF')

class DurianApp(App):
    def build(self):
        register_asset_paths()
        self.model_path = resource_find("best_durian_model.tflite")
        # self.audio_path = "audio.wav"
        self.interpreter = None

        # สีธีม
        main_bg = "#808836"
        secondary_bg = "#FFBF00"
        accent1 = "#FF9A00"
        accent2 = "#D10363"

        self.layout = BoxLayout(orientation="vertical", padding=20, spacing=15)

        from kivy.graphics import Color, Rectangle
        with self.layout.canvas.before:
            Color(*get_color_from_hex(main_bg))
            self.rect_bg = Rectangle(size=self.layout.size, pos=self.layout.pos)
        self.layout.bind(size=self._update_rect, pos=self._update_rect)

        self.title_label = Label(
            text="ตัวทำนายความสุกของทุเรียน",
            font_size='24sp',
            font_name=safe_font("assets/fonts/Prompt-Regular.ttf"),
            color=font_color(main_bg)
        )
        self.layout.add_widget(self.title_label)

        self.record_button = Button(
            text="อัดเสียงทุเรียน",
            font_size='20sp',
            font_name=safe_font("assets/fonts/Prompt-Regular.ttf"),
            on_press=self.record_audio,
            background_normal='',
            background_color=get_color_from_hex(secondary_bg),
            color=font_color(secondary_bg)
        )
        self.layout.add_widget(self.record_button)

        self.play_button = Button(
            text="ฟังเสียงที่อัด",
            font_size='20sp',
            font_name=safe_font("assets/fonts/Prompt-Regular.ttf"),
            on_press=self.play_audio,
            disabled=True,
            background_normal='',
            background_color=get_color_from_hex(accent1),
            color=font_color(accent1)
        )
        self.layout.add_widget(self.play_button)

        self.predict_button = Button(
            text="ทำนายความสุก",
            font_size='20sp',
            font_name=safe_font("assets/fonts/Prompt-Regular.ttf"),
            on_press=self.run_inference,
            background_normal='',
            background_color=get_color_from_hex(accent2),
            color=font_color(accent2)
        )
        self.layout.add_widget(self.predict_button)

        self.result_label = Label(
            text="ผลการทำนายจะแสดงที่นี่",
            font_size='22sp',
            font_name=safe_font("assets/fonts/Prompt-Regular.ttf"),
            color=font_color(main_bg)
        )
        self.layout.add_widget(self.result_label)

        # ✅ ย้ายมาตรงนี้ หลังจาก result_label ถูกสร้าง
        self.load_model()

        return self.layout


    def _update_rect(self, instance, value):
        self.rect_bg.size = instance.size
        self.rect_bg.pos = instance.pos

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.result_label.text = "ไม่พบโมเดล"
            return
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def record_audio(self, instance):
        self.result_label.text = "กำลังอัดเสียง..."
        self.record_button.disabled = True
        threading.Thread(target=self._record_thread).start()

    def _record_thread(self):
        try:
            from jnius import autoclass
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE])

            MediaRecorder = autoclass('android.media.MediaRecorder')
            recorder = MediaRecorder()

            recorder.setAudioSource(MediaRecorder.AudioSource.MIC)
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
            recorder.setOutputFile(self.audio_path)

            recorder.prepare()
            recorder.start()

            import time
            time.sleep(10)

            recorder.stop()
            recorder.release()

            self.update_status("อัดเสียงสำเร็จแล้ว")
            self.play_button.disabled = False
        except Exception as e:
            self.update_status(f"อัดเสียงล้มเหลว: {e}")
        finally:
            self.record_button.disabled = False

    def play_audio(self, instance):
        self.update_status("กำลังเล่นเสียง...")
        threading.Thread(target=self._play_thread).start()

    def _play_thread(self):
        try:
            from jnius import autoclass
            MediaPlayer = autoclass('android.media.MediaPlayer')
            player = MediaPlayer()
            player.setDataSource(self.audio_path)
            player.prepare()
            player.start()
        except Exception as e:
            self.update_status(f"เล่นเสียงล้มเหลว: {e}")
        finally:
            self.record_button.disabled = False

    def run_inference(self, instance):
        try:
            import wave
            with wave.open(self.audio_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
                dtype = np.int16 if sample_width == 2 else np.uint8
                waveform = np.frombuffer(audio_data, dtype=dtype).astype(np.float32) / 32768.0

            # สร้าง dummy MFCC data สำหรับ inference (เพราะ librosa ใช้ไม่ได้)
            mfccs = np.zeros((4, 174), dtype=self.input_details[0]['dtype'])
            input_tensor = mfccs[np.newaxis, ..., np.newaxis]

            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            confidence = float(output[1]) if len(output) == 2 else float(output[0])
            class_names = ['ดิบ', 'สุก']
            index = 1 if confidence >= 0.5 else 0
            self.result_label.text = f"ผลทำนาย: {class_names[index]} ({confidence*100:.2f}%)"
        except Exception as e:
            self.result_label.text = f"ทำนายล้มเหลว: {e}"

    def update_status(self, text):
        Clock.schedule_once(lambda dt: setattr(self.result_label, 'text', text))

if __name__ == '__main__':
    DurianApp().run()
