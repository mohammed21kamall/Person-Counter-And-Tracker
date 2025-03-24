import threading
import queue
import logging
import dlib
import pymysql
import numpy as np
import cv2
from datetime import datetime
from tracker.centroidtracker import CentroidTracker

class CameraProcess:
    def __init__(self, Branch, camera_config):
        self.Branch = Branch
        self.cam_config = camera_config
        self.window_name = f"Camera {self.cam_config['camera']}"  # Unique window name for each camera

        # Initialize logging
        logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
        self.logger = logging.getLogger(__name__)

        # Load model configurations
        self.prototxt_path = "detector/MobileNetSSD_deploy.prototxt"
        self.model_path = "detector/MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)

        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

        # Initialize detection classes
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # Initialize variables
        self.confidence_threshold = 0.6
        self.skip_frames = 20
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
        if self.cam_config:
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
        select_query = f"SELECT UserName, Password, IP, Channel, PeopleCounter, Heatmap, DirectionPeople, CameraNum FROM Camera WHERE Branch = {self.Branch}"
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
                "port": "554",
                "username": result[0],
                "password": result[1],
                "channel": result[3],
                "camera": result[7]
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

    def bulk_insert_tracking_data(self, tracking_data):
        """Bulk insert tracking data into the database, ignoring duplicates."""
        connection = self.connect_to_db()
        if connection is None:
            self.logger.error("Failed to connect to the database.")
            return

        try:
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO DirectionTest 
                (Branch, Camera, PersonID, X, Y, Direction, DateTime) 
                VALUES (%s, %s, %s, %s, %s, %s, %s) 
                ON DUPLICATE KEY UPDATE
                X = VALUES(X), Y = VALUES(Y), Direction = VALUES(Direction), DateTime = VALUES(DateTime)
                """
                cursor.executemany(sql, tracking_data)
                connection.commit()
                self.logger.info(f"Successfully inserted {len(tracking_data)} records.")
        except Exception as e:
            self.logger.error(f"Bulk database insertion error: {e}")
        finally:
            connection.close()

    def process_frame(self, frame, net, ct, trackers):
        """Process a single frame for detection and tracking."""

        frame = cv2.resize(frame, (500, int(frame.shape[0] * (500 / frame.shape[1]))))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                    tracker.start_track(rgb, rect)  # Ensure `rgb` is always defined
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

            obj_date = datetime.now()
            self.trackable_objects[objectID] = {"centroid": centroid, "direction": direction, "date": obj_date}

            text = f"ID {objectID}, {direction}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        return frame, status, rects

    def merged_capture_process_thread(self, vs):
        """Thread for processing frames..."""
        trackers = []
        tracking_data_batch = []  # List to hold batch data
        frame_counter = 0  # Initialize frame counter
        interval_frames = 128  # 10 seconds at 25 fps

        self.logger.info("Frame processing thread is running.")

        while self.is_running:
            try:
                ret, frame = vs.read()
                if not ret:
                    self.logger.error("Failed to capture frame")
                    self.is_running = False
                    break

                # Process frame
                processed_frame, status, _ = self.process_frame(frame, self.net, self.ct, trackers)

                self.totalFrames += 1
                frame_counter += 1  # Increment frame counter
                # Add to processed queue
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put((processed_frame, status))

                # Collect direction data every 10 seconds (250 frames)
                if frame_counter >= interval_frames:
                    self.logger.info(f"10 seconds ({interval_frames} frames) have passed. Collecting direction data.")
                    self.logger.info(f"Trackable objects: {self.trackable_objects}")

                    # Accumulate tracking data in the batch list
                    for objectID, data in self.trackable_objects.items():
                        direction = data["direction"]
                        centroid = data["centroid"]
                        obj_date = data["date"]

                        tracking_data_batch.append((
                            self.Branch,
                            self.cam_config["camera"],
                            objectID,
                            centroid[0],
                            centroid[1],
                            direction,
                            obj_date
                        ))

                    # Insert data into the database and clear the batch list
                    if tracking_data_batch:  # Check if there is data to insert
                        self.logger.info(f"Inserting {len(tracking_data_batch)} records into the database.")
                        self.bulk_insert_tracking_data(tracking_data_batch)
                        tracking_data_batch.clear()  # Clear the batch list after insertion
                        self.logger.info("Batch list cleared after database insertion.")

                    # Reset the frame counter for the next interval
                    frame_counter = 0
                    self.trackable_objects.clear()
                continue
            except Exception as e:
                self.logger.error(f"Error in frame processing: {e}")

    def insert_image_to_db(self, image):
        """Insert an image into the database, avoiding duplicates for Branch and Camera."""
        connection = self.connect_to_db()
        if connection is None:
            self.logger.error("Failed to connect to the database.")
            return
        try:
            with connection.cursor() as cursor:
                # Convert the image to a binary format
                _, img_encoded = cv2.imencode('.jpg', image)
                img_binary = img_encoded.tobytes()

                # SQL query to insert into the table
                sql = """
                INSERT INTO captured_images 
                (Branch, Camera, image) 
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                image = VALUES(image)  -- Update the image if duplicate
                """
                # Execute the query with the appropriate values
                cursor.execute(sql, (
                    self.Branch,  # Branch value
                    self.cam_config["camera"],  # Camera value
                    img_binary  # Image binary data
                ))
                connection.commit()
                self.logger.info(
                    f"Successfully inserted/updated image for Branch {self.Branch}, Camera {self.cam_config['camera']}.")
        except Exception as e:
            self.logger.error(f"Database insertion error: {e}")
        finally:
            connection.close()

    def display_thread(self):
        """Thread for displaying processed frames and capturing an image once."""
        image_captured = False  # Flag to track if the image has been captured
        while self.is_running:
            try:
                frame, status = self.processed_frame_queue.get(timeout=1)

                # Display the frame
                cv2.putText(frame, f"Status: {status}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(self.window_name, frame)

                # Capture and insert the image once
                if not image_captured:
                    timestamp = datetime.now()  # Current timestamp
                    self.insert_image_to_db(frame)
                    image_captured = True  # Set the flag to True after capturing the image
                    self.logger.info("Image captured and inserted into the database.")

                # Check for user input to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.is_running = False
                    break
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in display thread: {e}")

    def start_detector_tracker(self):
        """Initialize and start combined thread"""
        try:
            vs = cv2.VideoCapture(self.build_rtsp_url())
            if not vs.isOpened():
                raise ValueError("Failed to open video stream")

            # Start combined thread
            main_thread = threading.Thread(
                target=self.merged_capture_process_thread,
                args=(vs,),
                daemon=True
            )
            main_thread.start()

            # Start display thread separately
            display_thread = threading.Thread(target=self.display_thread, daemon=True)
            display_thread.start()

            main_thread.join()
            display_thread.join()

        except Exception as e:
            self.logger.error(f"Error: {e}")
        finally:
            self.is_running = False
            vs.release()
            cv2.destroyWindow(self.window_name)
