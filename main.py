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
from kivy.resources import resource_find, resource_add_path

# ตรวจสอบว่าเป็น Android หรือไม่
is_android = platform.system() == 'Linux' and 'ANDROID_ARGUMENT' in os.environ

# เพิ่ม path assets (พยายามให้ถูกต้องก่อน)
resource_add_path(os.path.abspath("."))
resource_add_path(os.path.abspath(os.path.join("assets", "fonts")))
resource_add_path(os.path.abspath(os.path.join("assets", "models")))

if is_android:
    from android.permissions import request_permissions, Permission
    from android.storage import app_storage_path
    request_permissions([Permission.RECORD_AUDIO, Permission.WRITE_EXTERNAL_STORAGE])
    audio_dir = app_storage_path()
else:
    audio_dir = "."

# ใช้ TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ฟังก์ชัน fallback asset path
def find_asset(path):
    found = resource_find(path)
    if found:
        return found
    fallback_path = f"assets/{path}" if not path.startswith("assets/") else path
    found_fallback = resource_find(fallback_path)
    if found_fallback:
        print(f"[WARN] ใช้ fallback asset: {fallback_path}")
        return found_fallback
    print(f"[ERROR] ไม่พบ asset: {path}")
    return None

# ฟังก์ชันฟอนต์ fallback
def safe_font(path):
    if not path:
        print("[WARN] ฟอนต์ไม่พบ → ใช้ Roboto")
        return "Roboto"
    return path

# คำนวณสีข้อความจากพื้นหลัง
def font_color(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    return get_color_from_hex('#000000') if luminance > 0.5 else get_color_from_hex('#FFFFFF')

class DurianApp(App):
    def tr(self, th_text, en_text):
        return th_text if self.lang == 'th' else en_text

    def build(self):
        # โหลดโมเดลและฟอนต์
        self.model_path = find_asset("models/best_durian_model.tflite")
        self.audio_path = os.path.join(audio_dir, "audio.wav")
        self.font_path = safe_font(find_asset("fonts/Prompt-Regular.ttf"))
        self.lang = 'th' if self.font_path != "Roboto" else 'en'
        self.interpreter = None

        # สีธีม
        main_bg = "#808836"
        secondary_bg = "#FFBF00"
        accent1 = "#FF9A00"
        accent2 = "#D10363"

        # Layout หลัก
        self.layout = BoxLayout(orientation="vertical", padding=20, spacing=15)
        from kivy.graphics import Color, Rectangle
        with self.layout.canvas.before:
            Color(*get_color_from_hex(main_bg))
            self.rect_bg = Rectangle(size=self.layout.size, pos=self.layout.pos)
        self.layout.bind(size=self._update_rect, pos=self._update_rect)

        # Title
        self.title_label = Label(
            text=self.tr("ตัวทำนายความสุกของทุเรียน", "Durian Ripeness Classifier"),
            font_size='24sp',
            font_name=self.font_path,
            color=font_color(main_bg)
        )
        self.layout.add_widget(self.title_label)

        # ปุ่มอัดเสียง
        self.record_button = Button(
            text=self.tr("อัดเสียงทุเรียน", "Record Durian Sound"),
            font_size='20sp',
            font_name=self.font_path,
            on_press=self.record_audio,
            background_normal='',
            background_color=get_color_from_hex(secondary_bg),
            color=font_color(secondary_bg)
        )
        self.layout.add_widget(self.record_button)

        # ปุ่มเล่นเสียง
        self.play_button = Button(
            text=self.tr("ฟังเสียงที่อัด", "Play Recorded Sound"),
            font_size='20sp',
            font_name=self.font_path,
            on_press=self.play_audio,
            disabled=True,
            background_normal='',
            background_color=get_color_from_hex(accent1),
            color=font_color(accent1)
        )
        self.layout.add_widget(self.play_button)

        # ปุ่มทำนาย
        self.predict_button = Button(
            text=self.tr("ทำนายความสุก", "Predict Ripeness"),
            font_size='20sp',
            font_name=self.font_path,
            on_press=self.run_inference,
            background_normal='',
            background_color=get_color_from_hex(accent2),
            color=font_color(accent2)
        )
        self.layout.add_widget(self.predict_button)

        # ป้ายผลลัพธ์
        self.result_label = Label(
            text=self.tr("ผลการทำนายจะแสดงที่นี่", "Prediction will be shown here"),
            font_size='22sp',
            font_name=self.font_path,
            color=font_color(main_bg)
        )
        self.layout.add_widget(self.result_label)

        self.load_model()
        return self.layout

    def _update_rect(self, instance, value):
        self.rect_bg.size = instance.size
        self.rect_bg.pos = instance.pos

    def load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"[ERROR] ไม่พบโมเดล: {self.model_path}")
            self.result_label.text = self.tr("ไม่พบโมเดล", "Model not found")
            return
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def record_audio(self, instance):
        self.result_label.text = self.tr("กำลังอัดเสียง...", "Recording...")
        self.record_button.disabled = True
        threading.Thread(target=self._record_thread).start()

    def _record_thread(self):
        try:
            from jnius import autoclass
            MediaRecorder = autoclass('android.media.MediaRecorder')
            recorder = MediaRecorder()
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC)
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
            recorder.setOutputFile(self.audio_path)

            recorder.prepare()
            recorder.start()
            import time; time.sleep(5)
            recorder.stop()
            recorder.release()

            self.update_status(self.tr("อัดเสียงสำเร็จแล้ว", "Recording complete"))
            self.play_button.disabled = False
        except Exception as e:
            self.update_status(self.tr(f"อัดเสียงล้มเหลว: {e}", f"Recording failed: {e}"))
            print("[ERROR] อัดเสียงล้มเหลว:", e)
        finally:
            self.record_button.disabled = False

    def play_audio(self, instance):
        self.update_status(self.tr("กำลังเล่นเสียง...", "Playing sound..."))
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
            self.update_status(self.tr(f"เล่นเสียงล้มเหลว: {e}", f"Playback failed: {e}"))
            print("[ERROR] เล่นเสียงล้มเหลว:", e)
        finally:
            self.record_button.disabled = False

    def run_inference(self, instance):
        try:
            import wave
            if not os.path.exists(self.audio_path):
                self.result_label.text = self.tr("ไม่พบไฟล์เสียง", "Audio file not found")
                return

            with wave.open(self.audio_path, 'rb') as wav_file:
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
                dtype = np.int16 if sample_width == 2 else np.uint8
                waveform = np.frombuffer(audio_data, dtype=dtype).astype(np.float32) / 32768.0

            mfccs = np.zeros((4, 174), dtype=self.input_details[0]['dtype'])  # ปรับตามโมเดล
            input_tensor = mfccs[np.newaxis, ..., np.newaxis]

            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            confidence = float(output[1]) if len(output) == 2 else float(output[0])
            class_names = [self.tr("ดิบ", "Unripe"), self.tr("สุก", "Ripe")]
            index = 1 if confidence >= 0.5 else 0
            self.result_label.text = self.tr(
                f"ผลทำนาย: {class_names[index]} ({confidence*100:.2f}%)",
                f"Prediction: {class_names[index]} ({confidence*100:.2f}%)"
            )
        except Exception as e:
            self.result_label.text = self.tr(f"ทำนายล้มเหลว: {e}", f"Inference failed: {e}")
            print("[ERROR] ทำนายล้มเหลว:", e)

    def update_status(self, text):
        Clock.schedule_once(lambda dt: setattr(self.result_label, 'text', text))

if __name__ == '__main__':
    DurianApp().run()
