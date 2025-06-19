from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import numpy as np
import os
import sounddevice as sd
import soundfile as sf
import librosa
import threading
from kivy.clock import Clock
from kivy.utils import get_color_from_hex
from scipy.signal import butter, lfilter

# โหลด TFLite Interpreter
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("Using tensorflow.lite")

def font_color(hex_color):
    """เลือกสีฟอนต์: ดำถ้าพื้นสว่าง, ขาวถ้าพื้นเข้ม"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    return get_color_from_hex('#000000') if luminance > 0.5 else get_color_from_hex('#FFFFFF')

class DurianApp(App):
    def build(self):
        self.model_path = "./best_durian_model.tflite"
        self.audio_path = "audio.wav"

        # โหลดโมเดล
        self.interpreter = None
        self.load_model()

        # กำหนดสีธีม
        main_bg = "#808836"       # สีพื้นหลัก
        secondary_bg = "#FFBF00"  # สีรอง
        accent1 = "#FF9A00"       # สีเสริม1
        accent2 = "#D10363"       # สีเสริม2

        self.layout = BoxLayout(orientation="vertical", padding=20, spacing=15)

        # กำหนดพื้นหลังของ layout ด้วย canvas
        with self.layout.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(*get_color_from_hex(main_bg))
            self.rect_bg = Rectangle(size=self.layout.size, pos=self.layout.pos)

        def update_rect(instance, value):
            self.rect_bg.size = instance.size
            self.rect_bg.pos = instance.pos

        self.layout.bind(size=update_rect, pos=update_rect)

        # title label
        self.title_label = Label(
            text="ตัวทำนายความสุกของทุเรียน",
            font_size='24sp',
            font_name="Prompt-Regular.ttf",
            color=font_color(main_bg)  # สีฟอนต์ขาว เพราะพื้นเข้ม
        )
        self.layout.add_widget(self.title_label)

        # ปุ่มอัดเสียง
        self.record_button = Button(
            text="อัดเสียงทุเรียน",
            font_size='20sp',
            font_name="Prompt-Regular.ttf",
            on_press=self.record_audio,
            background_normal='',
            background_color=get_color_from_hex(secondary_bg),
            color=font_color(secondary_bg)
        )
        self.layout.add_widget(self.record_button)

        # ปุ่มฟังเสียง
        self.play_button = Button(
            text="ฟังเสียงที่อัด",
            font_size='20sp',
            font_name="Prompt-Regular.ttf",
            on_press=self.play_audio,
            disabled=True,
            background_normal='',
            background_color=get_color_from_hex(accent1),
            color=font_color(accent1)
        )
        self.layout.add_widget(self.play_button)

        # ปุ่มทำนาย
        self.predict_button = Button(
            text="ทำนายความสุก",
            font_size='20sp',
            font_name="Prompt-Regular.ttf",
            on_press=self.run_inference,
            background_normal='',
            background_color=get_color_from_hex(accent2),
            color=font_color(accent2)
        )
        self.layout.add_widget(self.predict_button)

        # label แสดงผล
        self.result_label = Label(
            text="ผลการทำนายจะแสดงที่นี่",
            font_size='22sp',
            font_name="Prompt-Regular.ttf",
            color=font_color(main_bg)
        )
        self.layout.add_widget(self.result_label)

        return self.layout

    def load_model(self):
        if not os.path.exists(self.model_path):
            print("ไม่พบ model.tflite")
            return
        try:
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # ✅ ตรวจสอบ shape ที่โมเดลต้องการ
            print("Input shape from model:", self.input_details[0]['shape'])

            print("โหลดโมเดลสำเร็จ")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = DurianApp.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def record_audio(self, instance):
        import librosa.effects

        fs = 22050
        duration = 10  # 10 วินาที
        self.result_label.text = "กำลังอัดเสียง..."
        self.record_button.disabled = True  # ปิดปุ่มอัดเสียงระหว่างอัด

        def _record_thread():
            try:
                audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()

                audio = audio.flatten()  # แปลงเป็น array 1D

                # ลด noise ด้วย bandpass filter (กำหนดช่วง 300 - 5000 Hz)
                audio = self.bandpass_filter(audio, lowcut=300, highcut=5000, fs=fs, order=4)

                # Normalize
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.9

                # เพิ่ม pre-emphasis (เน้นความถี่สูง)
                audio = librosa.effects.preemphasis(audio, coef=0.97)

                # เพิ่ม gain 3 เท่า
                gain = 1
                boosted_audio = audio * gain

                # Clip ไม่ให้เกิน -1 ถึง 1
                boosted_audio = np.clip(boosted_audio, -1.0, 1.0)

                # บันทึกไฟล์เสียง
                sf.write(self.audio_path, boosted_audio, fs)

                self.update_status("บันทึกเสียงเสร็จแล้ว (ลดเสียงรบกวน + เพิ่มความดัง)")
                self.play_button.disabled = False  # เปิดใช้งานปุ่มฟังเสียง
                self.debug_audio(self.audio_path)
            except Exception as e:
                self.update_status(f"เกิดข้อผิดพลาด: {e}")
                print(e)
            finally:
                self.record_button.disabled = False  # เปิดปุ่มอัดเสียงกลับมา

        threading.Thread(target=_record_thread).start()

    def debug_audio(self, path):
        try:
            y, sr = librosa.load(path, sr=22050)

            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.sqrt(np.mean(y**2))
            peak = np.max(np.abs(y))
            mean_freq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

            print(f"Debug Audio:")
            print(f" - ความยาวไฟล์: {duration:.2f} วินาที")
            print(f" - RMS (ความดังเฉลี่ย): {rms:.5f}")
            print(f" - Peak (ความดังสูงสุด): {peak:.5f}")
            print(f" - ความถี่เฉลี่ย (Spectral Centroid): {mean_freq:.2f} Hz")

            # เช็คเสียงเบาเกินไป
            threshold = 0.01
            if rms < threshold:
                self.update_status("เสียงเบามาก กรุณาอัดให้ชัดขึ้น")
        except Exception as e:
            print(f"Debug error: {e}")

    def play_audio(self, instance):
        threading.Thread(target=self._play_audio_thread).start()

    def _play_audio_thread(self):
        try:
            self.update_status("กำลังเล่นเสียง...")
            data, fs = sf.read(self.audio_path, dtype='float32')
            sd.play(data, fs)
            self.record_button.disabled = True  # ปิดใช้งานปุ่มอัดเสียง
            sd.wait()  # รอจนเล่นเสียงเสร็จ
            self.update_status("เล่นเสียงเสร็จแล้ว")
        except Exception as e:
            self.update_status(f"เกิดข้อผิดพลาดขณะเล่นเสียง: {e}")
            print(e)
        finally:
            self.record_button.disabled = False  # เปิดปุ่มอัดเสียงกลับมา

    def update_status(self, text):
        Clock.schedule_once(lambda dt: setattr(self.result_label, 'text', text))

    def run_inference(self, instance):
        if self.interpreter is None:
            self.result_label.text = "ยังไม่ได้โหลดโมเดล"
            return
        if not os.path.exists(self.audio_path):
            self.result_label.text = "ยังไม่มีไฟล์เสียง"
            return
        try:
            y, sr = librosa.load(self.audio_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
            max_len = 174
            if mfccs.shape[1] < max_len:
                pad_width = max_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_len]

            mfccs = mfccs[np.newaxis, ..., np.newaxis].astype(self.input_details[0]['dtype'])

            print("MFCC shape:", mfccs.shape)
            print("MFCC sample data (first 5 coefficients, first 5 frames):")
            print(mfccs[0, :5, :5, 0])

            self.interpreter.set_tensor(self.input_details[0]['index'], mfccs)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            print("Raw output from model:", output)
            
            # ตรวจสอบ output shape
            if len(output) == 1:
                # Sigmoid output - probability of 'สุก'
                confidence = float(output[0])
            elif len(output) == 2:
                # Softmax output - output[1] คือ 'สุก'
                confidence = float(output[1])
            else:
                confidence = 0.0  # fallback

            class_names = ['ดิบ', 'สุก']  # 0 = ดิบ, 1 = สุก
            result_index = 1 if confidence >= 0.5 else 0
            result_text = f"ผลทำนาย: {class_names[result_index]} ({confidence*100:.2f}%)"

            self.result_label.text = result_text
            print(result_text)
        except Exception as e:
            self.result_label.text = f"เกิดข้อผิดพลาด: {e}"
            print(e)


if __name__ == '__main__':
    DurianApp().run()
