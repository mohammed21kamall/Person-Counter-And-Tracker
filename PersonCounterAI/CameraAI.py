from people_counter_with_threads import CameraProcess
import json

# Load branch information
with open("../Branches/Branch.json", "r") as file:
    branch_num = json.load(file)
branch_value = branch_num["branch"]

# Initialize CameraProcess
Cam1 = CameraProcess(Branch=branch_value, WaitInsert=10)
