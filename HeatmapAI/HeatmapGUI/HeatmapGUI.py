from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QDateTimeEdit,
                             QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, QDateTime, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QPoint, QTimer
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor
import matplotlib
matplotlib.use('Agg')
from Heatmap import Heatmap
import os
import math

class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 80)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.dots = 8
        self.dot_size = 10
        self.radius = 30
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = QPoint(self.width() // 2, self.height() // 2)
        for i in range(self.dots):
            angle = math.radians(self.angle + (360 / self.dots) * i)
            x = center.x() + self.radius * math.cos(angle)
            y = center.y() + self.radius * math.sin(angle)
            opacity = (i / self.dots)
            color = QColor(0, 123, 255)  
            color.setAlphaF(opacity)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(int(x - self.dot_size / 2),
                                int(y - self.dot_size / 2),
                                self.dot_size,
                                self.dot_size)

    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def start(self):
        self.timer.start(50)

    def stop(self):
        self.timer.stop()

class FadeAnimation(QPropertyAnimation):
    def __init__(self, target, property_name, duration=1000, parent=None):
        super().__init__(parent)
        self.setTargetObject(target)
        self.setPropertyName(property_name)
        self.setDuration(duration)
        self.setEasingCurve(QEasingCurve.InOutQuad)

class HeatmapGenerator(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, date_str, branch, camera):
        super().__init__()
        self.date_str = date_str
        self.branch = branch
        self.camera = camera

    def run(self):
        try:
            for i in range(101):
                self.progress.emit(i)
                self.msleep(20)  
            heatmap = Heatmap(self.date_str, self.branch, self.camera)
            heatmap.create_heatmap_overlay()
            filename_date = self.date_str.replace(' ', '_').replace(':', '-')
            output_file = os.path.join("../HeatmapAI/Heatmap", f"Heatmap_{filename_date}.png")
            if os.path.exists(output_file):
                self.finished.emit(output_file)
            else:
                self.error.emit("Heatmap file not generated")
        except Exception as e:
            self.error.emit(str(e))

class AnimatedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opacity = 1.0
    def setOpacity(self, opacity):
        self._opacity = opacity
        self.update()
    def opacity(self):
        return self._opacity
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        super().paintEvent(event)

class HeatmapGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heatmap Generator")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("Icon/heatmap.ico"))
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        controls_layout = QHBoxLayout()
        self.date_selector = QDateTimeEdit(self)
        self.date_selector.setDateTime(QDateTime.currentDateTime())
        self.date_selector.setDisplayFormat("yyyy-MM-dd")
        controls_layout.addWidget(QLabel("Select Date:"))
        controls_layout.addWidget(self.date_selector)
        self.branch_number_input = QLineEdit(self)
        self.branch_number_input.setPlaceholderText("Enter Branch Number")
        controls_layout.addWidget(QLabel("Branch Number:"))
        controls_layout.addWidget(self.branch_number_input)
        self.camera_number_input = QLineEdit(self)
        self.camera_number_input.setPlaceholderText("Enter Camera Number")
        controls_layout.addWidget(QLabel("Camera Number:"))
        controls_layout.addWidget(self.camera_number_input)
        self.generate_btn = QPushButton("Generate Heatmap")
        self.generate_btn.clicked.connect(self.generate_heatmap)
        controls_layout.addWidget(self.generate_btn)
        layout.addLayout(controls_layout)
        self.spinner = LoadingSpinner(self)
        self.spinner.hide()
        spinner_layout = QHBoxLayout()
        spinner_layout.addStretch()
        spinner_layout.addWidget(self.spinner)
        spinner_layout.addStretch()
        layout.addLayout(spinner_layout)
        self.image_label = AnimatedLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Heatmap will be displayed here")
        layout.addWidget(self.image_label)
        self.statusBar().showMessage("Ready")
        self.fade_out = FadeAnimation(self.image_label, b"opacity")
        self.fade_out.setStartValue(1.0)
        self.fade_out.setEndValue(0.0)
        self.fade_in = FadeAnimation(self.image_label, b"opacity")
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)

    def generate_heatmap(self):
        date_str = self.date_selector.dateTime().toString("yyyy-MM-dd")
        branch = self.branch_number_input.text()
        camera = self.camera_number_input.text()
        self.fade_out.start()
        self.spinner.show()
        self.spinner.start()
        self.generate_btn.setEnabled(False)
        self.worker = HeatmapGenerator(date_str, branch, camera)
        self.worker.finished.connect(self.on_heatmap_generated)
        self.worker.error.connect(self.on_heatmap_error)
        self.worker.progress.connect(self.update_status)
        self.statusBar().showMessage(f"Generating heatmap for {date_str}...")
        self.worker.start()
      
    def update_status(self, value):
        self.statusBar().showMessage(f"Generating heatmap... {value}%")
      
    def on_heatmap_generated(self, output_file):
        self.spinner.stop()
        self.spinner.hide()
        pixmap = QPixmap(output_file)
        scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.fade_in.start()
        self.generate_btn.setEnabled(True)
        self.statusBar().showMessage("Heatmap generated successfully")
      
    def on_heatmap_error(self, error_message):
        self.spinner.stop()
        self.spinner.hide()
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to generate heatmap: {error_message}")
        self.statusBar().showMessage("Error generating heatmap")
