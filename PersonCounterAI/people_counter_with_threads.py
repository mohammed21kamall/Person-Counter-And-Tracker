from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import pymysql
import threading
import queue
import json
import cv2
import logging
import time
import datetime
import dlib
import csv

class CameraProcess:
    def __init__(self, Branch, WaitInsert, camera_config):
        logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
        self.logger = logging.getLogger(__name__)
        self.Branch = Branch
        self.WaitInsert = WaitInsert
        self.cam = camera_config
        self.window_name = f"Camera {self.cam['camera']}"  
        self.load_configs()
        self.prototxt_path = "detector/MobileNetSSD_deploy.prototxt"
        self.model_path = "detector/MobileNetSSD_deploy.caffemodel"
        self.confidence_threshold = 0.4
        self.skip_frames = 2
        self.start_time = time.time()
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.W = None
        self.H = None
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
        self.total = []
        self.move_out = []
        self.move_in = []
        self.out_time = []
        self.in_time = []
        self.DisplayWriteOnFrame = False

        self.frame_queue = queue.Queue(maxsize=30)
        self.processed_frame_queue = queue.Queue(maxsize=30)
        self.alert_queue = queue.Queue()

        self.is_running = True
        self.processing_lock = threading.Lock()


        self.Person_Counter()

    def load_configs(self):
        """Load configuration files"""
        with open("utils/config.json", "r") as file:
            self.config = json.load(file)


    def build_rtsp_url(self):
        """Fetch camera configuration"""
        if self.cam:
            return f"rtsp://{self.cam['username']}:{self.cam['password']}@{self.cam['ip']}:{self.cam['port']}/Streaming/Channels/{self.cam['channel']}"
        else:
            return None


    def connect_to_db(self):
        """Establish a connection to the database."""
        return pymysql.connect(
            host="serastores.com",
            database="FlowAnalyticsDB",
            user="flow_admin",
            password="D4ta$Flow&2024"
        )

    def frame_capture_thread(self, vs):
        """Thread for capturing frames from the video stream"""
        while self.is_running:
            ret, frame = vs.read()
            if not ret:
                self.is_running = False
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                continue

    def log_data_thread(self):
        """Thread for logging data to the database"""
        while self.is_running:
            time.sleep(self.WaitInsert * 60)
            with self.processing_lock:
                self.log_data()

    def log_data(self):
        """Log people counting data to the database and save to a CSV file, then reset counters."""
        if not self.move_in and not self.move_out:
            return

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_file = "PersonCounterAI/person_counter_log.csv"

        try:
            connection = self.connect_to_db()
            cursor = connection.cursor()

            sql_insert_query = """INSERT INTO PersonCounter (Branch, Camera, MoveCount, Time, `IN/OUT`) 
                                VALUES (%s, %s, %s, %s, %s)"""

            csv_rows = []

            if self.in_time:
                data_tuple = (self.Branch, self.cam['camera'], len(self.move_in), current_time, 'in')
                cursor.execute(sql_insert_query, data_tuple)
                csv_rows.append(data_tuple)

            if self.out_time:
                data_tuple = (self.Branch, self.cam['camera'], len(self.move_out), current_time, 'out')
                cursor.execute(sql_insert_query, data_tuple)
                csv_rows.append(data_tuple)

            connection.commit()
            self.logger.info("Data successfully inserted into the PersonCounter table.")
            
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(['Branch', 'Camera', 'MoveCount', 'Time', 'IN/OUT'])
                writer.writerows(csv_rows)

            self.totalUp = 0
            self.totalDown = 0
            self.move_in.clear()
            self.move_out.clear()
            self.in_time.clear()
            self.out_time.clear()

        except pymysql.connect.Error as error:
            self.logger.error(f"Failed to insert data into MySQL table: {error}")
        except Exception as csv_error:
            self.logger.error(f"Failed to save data to CSV file: {csv_error}")

    def process_frame(self, frame, net, ct, trackers, trackableObjects):
        frame = cv2.resize(frame, (500, int(frame.shape[0] * (500 / frame.shape[1]))))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]
        status = "Waiting"
        rects = []
        if self.totalFrames % self.skip_frames == 0:
            status = "Detecting"
            trackers.clear()
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    if self.CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))
        return frame, status, rects

    def frame_processing_thread(self, net, ct):
        """Thread for processing frames"""
        trackers = []
        trackableObjects = {}
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
                frame, status, rects = self.process_frame(frame, net, ct, trackers, trackableObjects)
                objects = ct.update(rects)
                frame, trackableObjects = self.Counter_Process(frame, status, objects, trackableObjects)
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put((frame, status))
                self.totalFrames += 1
            except queue.Empty:
                continue

    def Counter_Process(self, frame, status, objects, trackableObjects):
        cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 0, 0), 2)
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
                to.crossed_line = False
            else:
                y_positions = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y_positions)
                to.centroids.append(centroid)
                if not to.counted:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if direction > 0 and centroid[1] > self.H // 2 and not to.crossed_line:
                        previous_y = y_positions[-1] if y_positions else 0
                        if previous_y < self.H // 2:
                            self.totalDown += 1
                            to.counted = True
                            to.crossed_line = True
                            self.move_in.append(self.totalDown)
                            self.in_time.append(current_time)
                    elif direction < 0 and centroid[1] < self.H // 2 and not to.crossed_line:
                        previous_y = y_positions[-1] if y_positions else self.H
                        if previous_y > self.H // 2:
                            self.totalUp += 1
                            to.counted = True
                            to.crossed_line = True
                            self.move_out.append(self.totalUp)
                            self.out_time.append(current_time)
            trackableObjects[objectID] = to
            if self.DisplayWriteOnFrame:
                text = f"ID {objectID}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        if self.DisplayWriteOnFrame:
            info_status = [
                ("Exit", self.totalUp),
                ("Enter", self.totalDown),
                ("Status", status),
            ]
            for (i, (k, v)) in enumerate(info_status):
                text = f"{k}: {v}"
                cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame, trackableObjects

    def display_thread(self):
        """Thread for displaying processed frames."""
        while self.is_running:
            try:
                frame, status = self.processed_frame_queue.get(timeout=1)
                cv2.putText(frame, f"Status: {status}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.is_running = False
                    break
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in display thread: {e}")

    def Person_Counter(self):
        self.logger.info("Starting the People Counter system...")
        net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        vs = cv2.VideoCapture(self.build_rtsp_url())
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        fps = FPS().start()
        threads = []
        capture_thread = threading.Thread(target=self.frame_capture_thread, args=(vs,))
        threads.append(capture_thread)
        processing_thread = threading.Thread(target=self.frame_processing_thread, args=(net, ct))
        threads.append(processing_thread)
        display_thread = threading.Thread(target=self.display_thread)
        threads.append(display_thread)
        db_thread = threading.Thread(target=self.log_data_thread)
        threads.append(db_thread)
        for thread in threads:
            thread.daemon = True
            thread.start()
        try:
            while self.is_running:
                fps.update()
                time.sleep(0.1) 
        except KeyboardInterrupt:
            self.is_running = False
        fps.stop()
        self.logger.info(f"Elapsed time: {fps.elapsed():.2f}")
        self.logger.info(f"Approx. FPS: {fps.fps():.2f}")
        for thread in threads:
            thread.join()
        cv2.destroyAllWindows()
        vs.release()
