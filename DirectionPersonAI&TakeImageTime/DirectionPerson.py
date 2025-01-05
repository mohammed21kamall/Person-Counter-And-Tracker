import threading
import queue
import logging
import dlib
import pymysql
import numpy as np
import cv2
from datetime import datetime
from tracker.centroidtracker import CentroidTracker
from imutils.video import FPS



class CameraProcess:
    def __init__(self, Branch):
        self.Branch = Branch
        self.cam_config = self.fetch_camera_config()

        # Initialize logging
        logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
        self.logger = logging.getLogger(__name__)

        # Load model configurations
        self.prototxt_path = "../detector/MobileNetSSD_deploy.prototxt"
        self.model_path = "../detector/MobileNetSSD_deploy.caffemodel"

        # Initialize detection classes
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # Initialize variables
        self.confidence_threshold = 0.4
        self.skip_frames = 30
        self.W = None
        self.H = None
        self.totalFrames = 0

        # Initialize queues for threading
        self.frame_queue = queue.Queue(maxsize=30)
        self.processed_frame_queue = queue.Queue(maxsize=30)

        # Threading control
        self.is_running = True

        # Trackable objects for direction logging
        self.trackable_objects = {}

        # Start detector and tracker
        if self.DirectionPeople == 1:
            self.start_detector_tracker()

    def connect_to_db(self):
        """Establish a connection to the database."""
        return pymysql.connect(
            host="serastores.com",
            database="FlowAnalyticsDB",
            user="flow_admin",
            password="D4ta$Flow&2024",
            autocommit=True
        )

    def fetch_camera_config(self):
        select_query = f"SELECT UserName, Password, IP, Channel, PeopleCounter, Heatmap, DirectionPeople FROM Camera WHERE Branch = {self.Branch}"
        connection = self.connect_to_db()
        cursor = connection.cursor()

        # Execute the query
        cursor.execute(select_query)

        # Fetch the data from the result
        result = cursor.fetchone()

        # If data is found, populate the dictionary
        if result:
            self.hikvision_config = {
                "ip": result[2],
                "port": "554",  # Assuming the port is fixed
                "username": result[0],
                "password": result[1],
                "channel": result[3]
            }

            self.DirectionPeople = result[6]
            return self.hikvision_config
        else:
            print("No data found.")
            return None

    def build_rtsp_url(self):
        """Construct RTSP URL from the camera configuration."""
        if not self.cam_config:
            raise ValueError("Camera configuration not available")
        return f"rtsp://{self.cam_config['username']}:{self.cam_config['password']}@{self.cam_config['ip']}:{self.cam_config['port']}/Streaming/Channels/{self.cam_config['channel']}"

    def insert_tracking_data(self, person_id, x, y, direction, timestamp):
        """Insert tracking data into the database."""
        connection = self.connect_to_db()
        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO PersonMovement 
                (PersonID, X, Y, Direction, DateTime) 
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (person_id, x, y, direction, timestamp))
        except Exception as e:
            self.logger.error(f"Database insertion error: {e}")
        finally:
            connection.close()

    def frame_capture_thread(self, vs):
        """Thread for capturing frames from the video stream."""
        while self.is_running:
            ret, frame = vs.read()
            if not ret:
                self.logger.error("Failed to capture frame")
                self.is_running = False
                break

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                continue

    def process_frame(self, frame, net, ct, trackers):
        """Process a single frame for detection and tracking."""
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

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            direction = "Stationary"

            if objectID in self.trackable_objects:
                prev_centroid = self.trackable_objects[objectID]["centroid"]
                delta_x = centroid[0] - prev_centroid[0]
                delta_y = centroid[1] - prev_centroid[1]

                if abs(delta_x) > abs(delta_y):
                    direction = "Right" if delta_x > 0 else "Left"
                elif abs(delta_y) > abs(delta_x):
                    direction = "Down" if delta_y > 0 else "Up"

            self.trackable_objects[objectID] = {"centroid": centroid, "direction": direction}

            text = f"ID {objectID}, {direction}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        return frame, status, rects

    def frame_processing_thread(self, net, ct):
        """Thread for processing frames."""
        trackers = []
        last_processed_time = datetime.now()
        last_frame = None

        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)

                frame, status, rects = self.process_frame(frame, net, ct, trackers)
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put((frame, status))

                last_frame = frame
                self.totalFrames += 1

                current_time = datetime.now()
                if (current_time - last_processed_time).total_seconds() >= 3:
                    last_processed_time = current_time

                    if last_frame is not None:
                        for objectID, data in self.trackable_objects.items():
                            direction = data["direction"]
                            centroid = data["centroid"]

                            self.insert_tracking_data(
                                person_id=objectID,
                                x=centroid[0],
                                y=centroid[1],
                                direction=direction,
                                timestamp=current_time
                            )
                        self.logger.info("Logged data to database for the last 3 seconds")

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in frame processing: {e}")

    def display_thread(self):
        """Thread for displaying processed frames."""
        while self.is_running:
            try:
                frame, status = self.processed_frame_queue.get(timeout=1)
                cv2.putText(frame, f"Status: {status}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Detector & Tracker", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.is_running = False
                    break
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in display thread: {e}")

    def start_detector_tracker(self):
        """Initialize and start detection and tracking."""
        self.logger.info("Starting the Detector & Tracker system...")

        try:
            net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
            vs = cv2.VideoCapture(self.build_rtsp_url())

            if not vs.isOpened():
                raise ValueError("Failed to open video stream")

            ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

            fps = FPS().start()

            threads = []

            capture_thread = threading.Thread(target=self.frame_capture_thread, args=(vs,))
            threads.append(capture_thread)

            processing_thread = threading.Thread(target=self.frame_processing_thread, args=(net, ct))
            threads.append(processing_thread)

            display_thread = threading.Thread(target=self.display_thread)
            threads.append(display_thread)

            for thread in threads:
                thread.daemon = True
                thread.start()

            for thread in threads:
                thread.join()

        except Exception as e:
            self.logger.error(f"Error in detector/tracker initialization: {e}")
        finally:
            self.is_running = False
            if 'vs' in locals():
                release = vs.release()
            cv2.destroyAllWindows()


