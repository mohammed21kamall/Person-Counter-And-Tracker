import json
import threading
from DirectionPerson import CameraProcess
from TakeImageFromCam import CaptureCam

with open("../Branches/Branch.json", "r") as file:
    branch_num = json.load(file)
branch_value = branch_num["branch"]

def run_camera_process(branch):
    detect = CameraProcess(Branch=branch)
    detect.run()

def run_capture_cam(branch, hour, minute, second):
    cap = CaptureCam(
        Branch=branch,
        CAPTURE_HOUR=hour,
        CAPTURE_MINUTE=minute,
        CAPTURE_SECOND=second,
    )
    cap.start()

# Set your desired capture time (24-hour format)
CAPTURE_HOUR = 23  # e.g., 13 for 1:00 PM
CAPTURE_MINUTE = 59  # e.g., 42 for 1:42 PM
CAPTURE_SECOND = 59  # e.g., 0 for 1:42:00

camera_thread = threading.Thread(target=run_camera_process, args=(branch_value,))
capture_thread = threading.Thread(
    target=run_capture_cam, args=(branch_value, CAPTURE_HOUR, CAPTURE_MINUTE, CAPTURE_SECOND)
)

camera_thread.start()
capture_thread.start()

camera_thread.join()
capture_thread.join()
