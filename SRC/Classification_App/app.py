import sys
import random
import cv2
from torchvision import transforms
import numpy as np
import tempfile
import os
import torch
from Crypto.Cipher import AES
from hashlib import sha256
from io import BytesIO
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QLineEdit, QMessageBox, QProgressBar, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize
from ultralytics import YOLO
from numba import njit
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
MODEL_DIR = REPO_ROOT / "Model" / "Classification"


def asset_path(name):
    return str(APP_DIR / "assets" / name)


table_file = MODEL_DIR / "nor_wli_nbi_table.npy"
try:
    nbi_table = np.load(table_file)
except Exception as e:
    print(f"Failed to load NBI table: {e}")
    nbi_table = np.zeros((256*256*256,), dtype=np.uint32)

def decrypt_model(encrypted_path, password):
    key = sha256(password.encode()).digest()
    with open(encrypted_path, 'rb') as f:
        data = f.read()
    iv = data[:16]
    ciphertext = data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext)
    pad_len = plaintext[-1]
    plaintext = plaintext[:-pad_len]
    return BytesIO(plaintext)

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

def update_class_thumbnails(self):
    for i, cname in enumerate(self.class_names):
        if self.class_collections[cname]:
            selected_img = random.choice(self.class_collections[cname])
            self.display_image(selected_img, self.class_thumbnails[i])

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.class_collections = {name: [] for name in model.names.values()}
        self.class_thumbnails = []
        self.class_labels = []
        self.class_names = list(model.names.values())
        self.class_frame_counters = {name: 0 for name in self.class_names}
        self.class_names = list(model.names.values())
        print(len(model.names))
        for cname in self.class_names:
            title = QLabel(cname)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 16px;")
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border: 2px solid #6c757d; background-color: white;")
            img_label.setMinimumSize(200, 150)
            img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.class_labels.append(title)
            self.class_thumbnails.append(img_label)
        self.initUI()
        self.writer_original = None
        self.writer_nbi = None
        self.writer_classification = None
        self.save_folder = None
        self.allow_saving = True
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
        
    def initUI(self):
        self.setWindowTitle('Spectrum Aided Vision Enhancer - Classification')
        self.setWindowIcon(QIcon(asset_path("icon.ico")))
        self.setStyleSheet("background-color: #F0F0F0; color: black;")
        dpi = self.logicalDpiX()
        scale_factor = dpi / 96
        font_size = int(18 * scale_factor)
        title_font_size = int(24 * scale_factor)
        self.label_title = QLabel('SPECTRUM AIDED VISION ENHANCER - CLASSIFICATION')
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setFont(QFont("Arial", title_font_size, QFont.Bold))
        self.label_title.setStyleSheet("padding: 10px; background-color: #D6D6D6; border-radius: 10px;")
        self.label_original_title = QLabel("White Light Video")
        self.label_nbi_title = QLabel("SAVE Video")
        self.label_detection_title = QLabel("Classification Result")
        for lbl, color in zip([self.label_original_title, self.label_nbi_title, self.label_detection_title], ["#1E90FF", "#32CD32", "#DC143C"]):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"font-size: {font_size}px; color: {color};")
        self.label_original = QLabel()
        self.label_nbi = QLabel()
        self.label_detection = QLabel()
        for lbl, color in zip([self.label_original, self.label_nbi, self.label_detection], ["#1E90FF", "#32CD32", "#DC143C"]):
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
        self.button_summary = QPushButton('Result Summary')
        self.button_summary.setIconSize(QSize(64, 64))
        self.button_summary.setStyleSheet(f"font-size: {font_size}px; background-color: #6c757d; color: white;")
        self.button_summary.setMaximumHeight(button_height)
        self.button_summary.setMaximumWidth(button_width1)
        self.button_summary.clicked.connect(self.show_result_summary)
        self.button_save.setMaximumHeight(button_height)
        self.button_save.setMaximumWidth(button_width1)
        self.button_load = QPushButton(QIcon(asset_path("upload.png")), 'Load Video')
        self.button_load.setIconSize(QSize(64, 64))
        self.button_load.setStyleSheet(f"font-size: {font_size}px; background-color: #17a2b8; color: white;")
        self.button_load.clicked.connect(self.load_media)
        self.button_load.setMaximumHeight(button_height)
        self.button_load.setMaximumWidth(button_width1)
        self.button_camera = QPushButton(QIcon(asset_path("camera.png")), 'Start Camera')
        self.button_camera.setIconSize(QSize(64, 64))
        self.button_camera.setStyleSheet(f"font-size: {font_size}px; background-color: #ffc107; color: black;")
        self.button_camera.clicked.connect(self.start_camera)
        self.button_camera.setMaximumHeight(button_height)
        self.button_camera.setMaximumWidth(button_width1)
        self.button_pause = QPushButton(QIcon(asset_path("pause.png")), 'Pause')
        self.button_pause.setIconSize(QSize(64, 64))
        self.button_pause.setStyleSheet(f"font-size: {font_size}px; background-color: #dc3545; color: white;")
        self.button_pause.clicked.connect(self.toggle_pause)
        self.button_pause.setMaximumHeight(button_height)
        self.button_pause.setMaximumWidth(button_width1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMaximumWidth(500)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setStyleSheet(f"font-size: {font_size}px; padding: 5px;")
        self.button_load_folder = QPushButton(
            QIcon(asset_path("folder.png")), 'Load Image Folder'
        )
        self.button_load_folder.setIconSize(QSize(64, 64))
        self.button_load_folder = QPushButton(
            QIcon(asset_path("folder.png")), 'Load Image Folder'
        )
        self.button_load_folder.setIconSize(QSize(64, 64))
        self.button_load_folder.setStyleSheet(f"font-size: {font_size}px; background-color: #17a2b8; color: white;")
        self.button_load_folder.clicked.connect(self.load_folder)
        self.button_load_folder.setMaximumHeight(button_height)
        self.button_load_folder.setMaximumWidth(button_width1)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.logo_label)
        control_layout.addWidget(self.patient_id_input)
        control_layout.addWidget(self.button_save)
        control_layout.addWidget(self.button_summary)
        control_layout.addWidget(self.button_load)
        control_layout.addWidget(self.button_load_folder)
        control_layout.addWidget(self.button_camera)
        control_layout.addWidget(self.button_pause)
        control_layout.addWidget(self.progress_bar)
        layout_original = QVBoxLayout()
        layout_nbi = QVBoxLayout()
        layout_detection = QVBoxLayout()
        for layout, title, label in zip([layout_original, layout_nbi, layout_detection], [self.label_original_title, self.label_nbi_title, self.label_detection_title], [self.label_original, self.label_nbi, self.label_detection]):
            layout.addWidget(title)
            layout.addWidget(label)
        frame_layout = QHBoxLayout()
        frame_layout.addLayout(layout_original)
        frame_layout.addLayout(layout_nbi)
        frame_layout.addLayout(layout_detection)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label_title)
        self.vbox.addLayout(frame_layout)
        self.class_display_layout = QVBoxLayout()
        thumb_row = QHBoxLayout()
        for title, img in zip(self.class_labels, self.class_thumbnails):
            col = QVBoxLayout()
            col.addWidget(title)
            col.addWidget(img)
            thumb_row.addLayout(col)
        self.class_display_layout.addLayout(thumb_row)
        self.vbox.addLayout(self.class_display_layout)
        self.vbox.addLayout(control_layout)
        self.setLayout(self.vbox)

    def classify_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)[0]
        if not hasattr(results, "probs") or results.probs is None:
            return image
        probs = results.probs.data.cpu().numpy()
        top_index = np.argmax(probs)
        top_class = model.names[top_index]
        confidence = probs[top_index]
        if confidence > 0.1:
            self.class_collections[top_class].append(image.copy())
        if self.save_folder and self.allow_saving:
            class_dir = os.path.join(self.save_folder, top_class)
            os.makedirs(class_dir, exist_ok=True)
            count = self.class_frame_counters[top_class]
            filename = f"frame_{count:04d}.png"
            cv2.imwrite(os.path.join(class_dir, filename), image)
            self.class_frame_counters[top_class] += 1
        annotated_img = image.copy()
        label = f"{top_class}: {confidence:.2%}"
        cv2.putText(annotated_img, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        self.update_class_thumbnails()  
        return annotated_img
        
    def setup_video_writers(self, width, height, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_id = self.patient_id_input.text().strip()
        if not patient_id:
            QMessageBox.warning(self, "Missing Info", "Please enter a Patient ID before processing the folder.")
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        base_folder = os.path.join(desktop, "SAVE Classification App Results")
        os.makedirs(base_folder, exist_ok=True)
        save_folder = os.path.join(base_folder, f"{timestamp}_{patient_id}")
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        self.writer_original = cv2.VideoWriter(os.path.join(save_folder, "original.avi"), fourcc, fps, (width, height))
        self.writer_nbi = cv2.VideoWriter(os.path.join(save_folder, "nbi.avi"), fourcc, fps, (width, height))
        self.writer_classification = cv2.VideoWriter(os.path.join(save_folder, "classification.avi"), fourcc, fps, (width, height))

    def release_writers(self):
        for writer in [self.writer_original, self.writer_nbi, self.writer_classification]:
            if writer:
                writer.release()
        self.writer_original = None
        self.writer_nbi = None
        self.writer_classification = None

    def update_class_thumbnails(self):
        for i, cname in enumerate(self.class_names):
            if self.class_collections[cname]:
                selected_img = random.choice(self.class_collections[cname])
                self.display_image(selected_img, self.class_thumbnails[i])

    def load_media(self):
        self.stop_camera()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image or Video File", "", "Images and Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv);;All Files (*)", options=options)
        if file_name:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                self.process_video(file_name)
            else:
                self.process_image(file_name)

    def process_video(self, file_path):
        patient_id = self.patient_id_input.text().strip()
        if not patient_id:
            QMessageBox.warning(self, "Missing Info", "Please enter a Patient ID before starting.")
            return
        self.cap = cv2.VideoCapture(file_path)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.setup_video_writers(width, height, int(fps))
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Failed to open video file.")
            return
        self.paused = False
        self.is_camera = False
        interval = int(1000 / fps) if fps > 0 else 33
        self.timer.start(interval)

    def start_camera(self):
        patient_id = self.patient_id_input.text().strip()
        if not patient_id:
            QMessageBox.warning(self, "Missing Info", "Please enter a Patient ID before starting.")
            return
        self.stop_camera()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open camera.")
            return
        self.paused = False
        self.is_camera = True
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.setup_video_writers(width, height, int(fps))
        interval = int(1000 / fps) if fps > 0 else 33
        self.timer.start(interval)

    def stop_camera(self):
        try:
            if self.cap:
                self.cap.release()
            self.timer.stop()
            self.progress_bar.setValue(0)
        finally:
            self.cap = None
        self.release_writers()

    def capture_frame(self):
        if self.paused or self.cap is None or not self.allow_saving:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        if not self.is_camera and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = int((pos / total) * 100)
            self.progress_bar.setValue(progress)
        nbi_frame = wli_to_nbi(frame, nbi_table)
        result_img = self.classify_image(nbi_frame)
        self.display_image(frame, self.label_original)
        self.display_image(nbi_frame, self.label_nbi)
        self.display_image(result_img, self.label_detection)
        if self.writer_original:
            self.writer_original.write(frame)
        if self.writer_nbi:
            self.writer_nbi.write(nbi_frame)
        if self.writer_classification:
            bgr_result = result_img if result_img.shape[2] == 3 else cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            self.writer_classification.write(bgr_result)

    def toggle_pause(self):
        self.paused = not self.paused
        self.button_pause.setText('Resume' if self.paused else 'Pause')
    
    def load_folder(self):
        self.stop_camera()
        patient_id = self.patient_id_input.text().strip()
        if not patient_id:
            QMessageBox.warning(self, "Missing Info", "Please enter a Patient ID before loading images.")
            return
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder", options=options)
        if folder_path:
            image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if not image_files:
                QMessageBox.warning(self, "No Images", "No valid images found in this folder.")
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            base_folder = os.path.join(desktop, "SAVE Classification App Results")
            os.makedirs(base_folder, exist_ok=True)
            save_folder = os.path.join(base_folder, f"{timestamp}_{patient_id}")
            os.makedirs(save_folder, exist_ok=True)
            self.save_folder = save_folder
            original_folder = os.path.join(save_folder, "original_images")
            nbi_folder = os.path.join(save_folder, "nbi_images")
            classified_folder = os.path.join(save_folder, "classified_images")
            os.makedirs(original_folder, exist_ok=True)
            os.makedirs(nbi_folder, exist_ok=True)
            os.makedirs(classified_folder, exist_ok=True)
            total = len(image_files)
            for idx, img_path in enumerate(sorted(image_files)):
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                nbi_frame = wli_to_nbi(frame, nbi_table)
                result_img = self.classify_image(nbi_frame)
                self.display_image(frame, self.label_original)
                self.display_image(nbi_frame, self.label_nbi)
                self.display_image(result_img, self.label_detection)
                basename = os.path.basename(img_path)
                base_no_ext = os.path.splitext(basename)[0]
                orig_pixmap = self.label_original.pixmap()
                if orig_pixmap:
                    orig_pixmap.save(os.path.join(original_folder, f"{base_no_ext}_original.png"))
                nbi_pixmap = self.label_nbi.pixmap()
                if nbi_pixmap:
                    nbi_pixmap.save(os.path.join(nbi_folder, f"{base_no_ext}_nbi.png"))
                class_pixmap = self.label_detection.pixmap()
                if class_pixmap:
                    class_pixmap.save(os.path.join(classified_folder, f"{base_no_ext}_classified.png"))
                self.progress_bar.setValue(int((idx + 1) / total * 100))
            QMessageBox.information(self, "Done", f"Processed {total} images. Results saved to:\n{self.save_folder}")

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

    def process_image(self, file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        nbi_frame = wli_to_nbi(frame, nbi_table)
        result_img = self.classify_image(nbi_frame)
        self.display_image(frame, self.label_original)
        self.display_image(nbi_frame, self.label_nbi)
        self.display_image(result_img, self.label_detection)

    def save_results(self):
        if not self.save_folder:
            QMessageBox.warning(self, "Save Error", "Recording folder not found. Start camera or video first.")
            return
        snapshot_path = os.path.join(self.save_folder, "classification_snapshot.png")
        pixmap = self.label_detection.pixmap()
        if pixmap:
            pixmap.save(snapshot_path)
        class_counts = {}
        total_images = 0
        for cname in self.class_names:
            class_dir = os.path.join(self.save_folder, cname)
            if os.path.exists(class_dir):
                count = len([
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                class_counts[cname] = count
                total_images += count
        if not class_counts:
            QMessageBox.information(self, "Saved", f"Snapshot saved to:\n{self.save_folder}\n\nNo classified images found.")
            return
        summary_lines = []
        for cname, count in class_counts.items():
            percent = (count / total_images) * 100 if total_images else 0
            summary_lines.append(f"{cname}: {count} images ({percent:.1f}%)")
        summary_text = "\n".join(summary_lines)
        QMessageBox.information(self, "Saved", f"Snapshot saved to:\n{self.save_folder}\n\nClassification Summary:\n{summary_text}")

    def show_result_summary(self):
        if not self.save_folder:
            QMessageBox.warning(self, "No Data", "No recorded data available. Please start camera or video first.")
            return
        self.timer.stop()
        self.allow_saving = False
        self.paused = True
        self.button_pause.setText("Resume")
        class_counts = {}
        total_images = 0
        for cname in self.class_names:
            class_dir = os.path.join(self.save_folder, cname)
            if os.path.exists(class_dir):
                count = len([
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                class_counts[cname] = count
                total_images += count
        if not class_counts:
            QMessageBox.information(self, "Summary", "No classified images found.")
            self.allow_saving = True
            self.timer.start(30)
            return
        summary_lines = []
        for cname, count in class_counts.items():
            percent = (count / total_images) * 100 if total_images else 0
            summary_lines.append(f"{cname}: {count} images ({percent:.1f}%)")
        summary_text = "\n".join(summary_lines)
        QMessageBox.information(self, "Classification Summary", f"Folder: {self.save_folder}\n\nSummary:\n{summary_text}")
        self.allow_saving = True
        self.timer.start(30)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    password, ok = QInputDialog.getText(
        None, "Model Password Required",
        "Enter password to unlock the model:",
        echo=QLineEdit.Password)
    if not ok or not password:
        QMessageBox.critical(None, "Access Denied", "No password provided. Exiting.")
        sys.exit(1)
    encrypted_model_file = MODEL_DIR / "best.pt.encrypted"
    try:
        decrypted_bytes = decrypt_model(encrypted_model_file, password)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(decrypted_bytes.read())
            tmp_path = tmp.name
        model = YOLO(tmp_path)
        print("Model loaded successfully.")
    except Exception as e:
        QMessageBox.critical(None, "Model Error", f"Failed to load model:\n{e}")
        sys.exit(1)
    window = ImageProcessor()
    window.showMaximized()
    sys.exit(app.exec_())
