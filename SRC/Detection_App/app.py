import sys
import cv2
import numpy as np
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QWidget, QSizePolicy, QLineEdit, QMessageBox, QProgressBar,
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize
from ultralytics import YOLO
from numba import njit
from datetime import datetime

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
MODEL_DIR = REPO_ROOT / "Model" / "Detection"


def asset_path(name):
    return str(APP_DIR / "assets" / name)


table_file = MODEL_DIR / "nor_wli_nbi_table.npy"
try:
    nbi_table = np.load(table_file)
except Exception as e:
    QMessageBox.warning(self, "Error", f"Failed to load NBI table: {e}")
nbi_table = np.load(table_file)

@njit
def fast_transfer(img, table):
    h, w, _ = img.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r, g, b = img[i, j]
            key = r + (g << 8) + (b << 16)
            val = table[key]
            out[i, j, 0] = val & 0xFF
            out[i, j, 1] = (val >> 8) & 0xFF
            out[i, j, 2] = (val >> 16) & 0xFF
    return out

def wli_to_nbi(wli, trans_table):
    image_rgb = cv2.cvtColor(wli, cv2.COLOR_BGR2RGB)
    pred_img2 = fast_transfer(image_rgb, trans_table)
    pred_img2 = cv2.cvtColor(pred_img2, cv2.COLOR_RGB2BGR)
    return pred_img2.astype('uint8')

def detect_cancer(image):
    results = model(image)
    r = results[0]
    class_info = []
    if r.boxes is not None:
        names = r.names
        boxes_to_keep = []
        for i, cls_id in enumerate(r.boxes.cls):
            class_name = names[int(cls_id)]
            if class_name in ["Dysplasia", "SCC"]:
                boxes_to_keep.append(i)
                conf = float(r.boxes.conf[i]) * 100
                class_info.append((class_name, conf))
        r.boxes = r.boxes[boxes_to_keep]
        return r.plot(), bool(r.boxes), class_info
    return image, False

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.detection_history = []
        self.cap = None
        self.paused = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.writer_original = None
        self.writer_nbi = None
        self.writer_detection = None
        self.recording = False
        self.recording_folder = ""
        self.is_camera = False
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.patient_id = ""
        self.initUI()

    def get_save_folder(self):
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        base_folder = os.path.join(desktop, "SAVE DETECTION RESULTS")
        os.makedirs(base_folder, exist_ok=True)
        folder = os.path.join(base_folder, f"{self.session_timestamp}_{self.patient_id}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def initUI(self):
        self.setWindowTitle('Spectrum Aided Vision Enhancer - Detection')
        self.setWindowIcon(QIcon(asset_path("icon.ico")))
        self.setStyleSheet("background-color: #F0F0F0; color: black;")
        dpi = self.logicalDpiX()
        scale_factor = dpi / 96
        font_size = int(18 * scale_factor)
        title_font_size = int(24 * scale_factor)
        self.label_title = QLabel('SPECTRUM AIDED VISION ENHANCER')
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setFont(QFont("Arial", title_font_size, QFont.Bold))
        self.label_title.setStyleSheet("padding: 10px; background-color: #D6D6D6; border-radius: 10px;")
        self.label_original_title = QLabel("White Light Video")
        self.label_nbi_title = QLabel("SAVE Video")
        self.label_detection_title = QLabel("Cancer Detection")
        for lbl, color in zip(
            [self.label_original_title, self.label_nbi_title, self.label_detection_title],
            ["#1E90FF", "#32CD32", "#DC143C"]):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"font-size: {font_size}px; color: {color};")
        self.label_original = QLabel()
        self.label_nbi = QLabel()
        self.label_detection = QLabel()
        for lbl, color in zip(
            [self.label_original, self.label_nbi, self.label_detection],
            ["#1E90FF", "#32CD32", "#DC143C"]):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"border: 3px solid {color}; background-color: white; padding: 10px;")
            lbl.setMinimumSize(640, 480)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button_height = 80
        button_width = 50
        button_width1 = 400
        self.logo_label = QLabel()
        pixmap = QPixmap(asset_path("logo.png"))
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setScaledContents(True)
        self.logo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.logo_label.setMinimumSize(130, 130)
        self.logo_label.setMaximumSize(130, 130)
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("Enter Patient ID")
        self.patient_id_input.setStyleSheet(f"font-size: {font_size}px; padding: 5px;")
        self.patient_id_input.setMaximumHeight(50)
        self.patient_id_input.setMaximumWidth(600)
        self.button_save = QPushButton(QIcon(asset_path("save.png")), 'SAVE')
        self.button_save.setIconSize(QSize(64, 64))
        self.button_save.setStyleSheet(f"font-size: {font_size}px; background-color: #28a745; color: white;")
        self.button_save.clicked.connect(self.save_results)
        self.button_save.setMaximumHeight(button_height)
        self.button_save.setMinimumWidth(button_width)
        self.button_save.setMaximumWidth(button_width1)
        self.button_load = QPushButton(QIcon(asset_path("upload.png")), 'Load Video')
        self.button_load.setIconSize(QSize(64, 64))
        self.button_load.setStyleSheet(f"font-size: {font_size}px; background-color: #17a2b8; color: white;")
        self.button_load.clicked.connect(self.load_media)
        self.button_load.setMaximumHeight(button_height)
        self.button_load.setMinimumWidth(button_width)
        self.button_load.setMaximumWidth(button_width1)
        self.button_camera = QPushButton(QIcon(asset_path("camera.png")), 'Start Camera')
        self.button_camera.setIconSize(QSize(64, 64))
        self.button_camera.setStyleSheet(f"font-size: {font_size}px; background-color: #ffc107; color: black;")
        self.button_camera.clicked.connect(self.start_camera)
        self.button_camera.setMaximumHeight(button_height)
        self.button_camera.setMinimumWidth(button_width)
        self.button_camera.setMaximumWidth(button_width1)
        self.button_pause = QPushButton(QIcon(asset_path("pause.png")), 'Pause')
        self.button_pause.setIconSize(QSize(64, 64))
        self.button_pause.setStyleSheet(f"font-size: {font_size}px; background-color: #dc3545; color: white;")
        self.button_pause.clicked.connect(self.toggle_pause)
        self.button_pause.setMaximumHeight(button_height)
        self.button_pause.setMinimumWidth(button_width)
        self.button_pause.setMaximumWidth(button_width1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMaximumWidth(500)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setStyleSheet(f"font-size: {font_size}px; padding: 5px;")
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.logo_label)
        control_layout.addWidget(self.patient_id_input)
        control_layout.addWidget(self.button_save)
        control_layout.addWidget(self.button_load)
        control_layout.addWidget(self.button_camera)
        control_layout.addWidget(self.button_pause)
        control_layout.addWidget(self.progress_bar)
        layout_original = QVBoxLayout()
        layout_nbi = QVBoxLayout()
        layout_detection = QVBoxLayout()
        for layout, title, label in zip(
            [layout_original, layout_nbi, layout_detection],
            [self.label_original_title, self.label_nbi_title, self.label_detection_title],
            [self.label_original, self.label_nbi, self.label_detection]):
            layout.addWidget(title)
            layout.addWidget(label)
        frame_layout = QHBoxLayout()
        frame_layout.addLayout(layout_original)
        frame_layout.addLayout(layout_nbi)
        frame_layout.addLayout(layout_detection)
        self.history_title = QLabel("Detected Cancer Snapshots")
        self.history_title.setAlignment(Qt.AlignCenter)
        self.history_title.setStyleSheet(f"font-size: {font_size + 2}px; padding: 5px;")
        self.history_labels = []
        self.history_layout = QHBoxLayout()
        for _ in range(5):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border: 2px solid #6c757d; background-color: #FFFFFF; padding: 5px;")
            lbl.setMinimumSize(200, 150)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.history_labels.append(lbl)
            self.history_layout.addWidget(lbl)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label_title)
        self.vbox.addLayout(frame_layout)
        self.vbox.addWidget(self.history_title)
        self.vbox.addLayout(self.history_layout)
        self.vbox.addLayout(control_layout)
        self.setLayout(self.vbox)

    def load_media(self):
        self.stop_camera()
        patient_id = self.patient_id_input.text().strip()
        if not patient_id:
            QMessageBox.warning(self, "Missing Info", "Please enter a Patient ID before loading media.")
            return
        self.patient_id = patient_id
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image or Video",
            "",
            "Images and Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_name:
            ext = os.path.splitext(file_name)[1].lower()
            print(f"Selected file: {file_name}")
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                self.process_video(file_name)
            else:
                self.process_image(file_name)
            return  
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select a Folder of Images"
        )
        if folder:
            print(f"Selected folder: {folder}")
            self.process_folder(folder)
        else:
            QMessageBox.information(self, "No Selection", "No file or folder was selected.")

    def process_image(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        nbi_img = wli_to_nbi(img, nbi_table)
        result_img, found_detection, class_info = detect_cancer(nbi_img)
        self.display_image(img, self.label_original)
        self.display_image(nbi_img, self.label_nbi)
        self.display_image(result_img, self.label_detection)
        if found_detection:
            self.add_to_detection_history(result_img)
        save_folder = self.get_save_folder()
        os.makedirs(save_folder, exist_ok=True)
        cv2.imwrite(os.path.join(save_folder, "detection_result.png"), result_img)
        QMessageBox.information(self, "Image Processed", f"Result saved to {save_folder}")
    
    def process_folder(self, folder_path):
        supported_ext = ['.png', '.jpg', '.jpeg', '.bmp']
        image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in supported_ext]
        if not image_files:
            QMessageBox.warning(self, "No Images", "No supported images found in the selected folder.")
            return
        save_folder = self.get_save_folder()
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        base_folder = os.path.join(desktop, "SAVE Classification App Results")
        os.makedirs(base_folder, exist_ok=True)
        save_folder = os.path.join(base_folder, f"{timestamp}_{patient_id}")
        os.makedirs(save_folder, exist_ok=True)
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {img_path} (could not read image)")
                continue
            nbi_img = wli_to_nbi(img, nbi_table)
            result_img, found_detection, class_info = detect_cancer(nbi_img)
            output_path = os.path.join(save_folder, f"detection_{img_name}")
            cv2.imwrite(output_path, result_img)
            self.display_image(img, self.label_original)
            self.display_image(nbi_img, self.label_nbi)
            self.display_image(result_img, self.label_detection)
            if found_detection:
                self.add_to_detection_history(result_img)
        QMessageBox.information(self, "Folder Processing Complete", f"Processed {len(image_files)} images.\nResults saved in {save_folder}.")
  
    def process_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Failed to open video file.")
            return
        self.setup_video_writers()
        self.paused = False
        self.is_camera = False
        self.timer.start(30)

    def start_camera(self):
        self.stop_camera()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open camera.")
            return
        self.setup_video_writers()
        self.paused = False
        self.is_camera = True
        self.timer.start(30)

    def setup_video_writers(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.recording_folder = self.get_save_folder()
        self.writer_original = cv2.VideoWriter(os.path.join(self.recording_folder, "original.avi"), fourcc, fps, (width, height))
        self.writer_nbi = cv2.VideoWriter(os.path.join(self.recording_folder, "nbi.avi"), fourcc, fps, (width, height))
        self.writer_detection = cv2.VideoWriter(os.path.join(self.recording_folder, "detection.avi"), fourcc, fps, (width, height))
        self.recording = True

    def cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.writer_original:
            self.writer_original.release()
            self.writer_original = None
        if self.writer_nbi:
            self.writer_nbi.release()
            self.writer_nbi = None
        if self.writer_detection:
            self.writer_detection.release()
            self.writer_detection = None
        self.recording = False

    def stop_camera(self):
        try:
            if self.cap:
                self.cap.release()
            self.timer.stop()
            self.progress_bar.setValue(0)
        finally:
            self.cleanup() 

    def capture_frame(self):
        if self.paused or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.release_writers()
            return
        if not self.is_camera and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = int((pos / total) * 100)
            self.progress_bar.setValue(progress)
        nbi_frame = wli_to_nbi(frame, nbi_table)
        result_img, found_detection, class_info = detect_cancer(nbi_frame)
        self.display_image(frame, self.label_original)
        self.display_image(nbi_frame, self.label_nbi)
        self.display_image(result_img, self.label_detection)
        if found_detection:
            self.add_to_detection_history(result_img)
        if self.recording:
            self.writer_original.write(frame)
            self.writer_nbi.write(nbi_frame)
            bgr_result = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR) if len(result_img.shape) == 3 else result_img
            self.writer_detection.write(bgr_result)

    def toggle_pause(self):
        self.paused = not self.paused
        self.button_pause.setText('Resume' if self.paused else 'Pause')

    def add_to_detection_history(self, image):
        self.detection_history.insert(0, image)
        self.detection_history = self.detection_history[:5]
        for i, frame in enumerate(self.detection_history):
            self.display_image(frame, self.history_labels[i])

    def display_image(self, img, label):
        if img is None:
            return
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        height, width, channel = img_rgb.shape
        bytes_per_line = channel * width
        qimg = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_results(self):
        self.release_writers()
        patient_id = self.patient_id_input.text().strip()
        if not patient_id:
            QMessageBox.warning(self, "Missing Info", "Please enter a Patient ID.")
            return
        final_folder = self.get_save_folder()
        if self.recording_folder and os.path.exists(self.recording_folder):
            for video in ["original.avi", "nbi.avi", "detection.avi"]:
                src = os.path.join(self.recording_folder, video)
                dst = os.path.join(final_folder, video)
                if os.path.exists(src):
                    os.rename(src, dst)
        snapshots_folder = os.path.join(final_folder, "snapshots")
        os.makedirs(snapshots_folder, exist_ok=True)
        for idx, img in enumerate(self.detection_history):
            path = os.path.join(snapshots_folder, f"snapshot_{idx+1}.png")
            cv2.imwrite(path, img)
        QMessageBox.information(self, "Saved", f"Results saved to {final_folder}")
        self.recording_folder = None

    def release_writers(self):
        if self.writer_original:
            self.writer_original.release()
            self.writer_original = None
        if self.writer_nbi:
            self.writer_nbi.release()
            self.writer_nbi = None
        if self.writer_detection:
            self.writer_detection.release()
            self.writer_detection = None
        self.recording = False

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        model = YOLO(MODEL_DIR / "best.pt")
        print("Model loaded successfully.")
    except Exception as e:
        QMessageBox.critical(None, "Model Error", f"Failed to load model:\n{e}")
        sys.exit(1)
    window = ImageProcessor()
    window.showMaximized()
    sys.exit(app.exec_())
