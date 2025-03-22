from people_counter_with_threads import CameraProcess
import json
import pymysql
import threading
with open("Branches/Branch.json", "r") as file:
    branch_num = json.load(file)
branch_value = branch_num["branch"]

def fetch_all_cameras_for_branch(branch):
    """Fetch all cameras for a specific branch."""
    connection = pymysql.connect(
        host="host.com",
        database="NameDB",
        user="your_user",
        password="password"
    )
    cursor = connection.cursor()
    select_query = f"SELECT UserName, Password, IP, CameraNum, channel FROM Camera WHERE Branch = {branch} AND PeopleCounter = 1"
    cursor.execute(select_query)
    results = cursor.fetchall()
    connection.close()
    cameras = []
    for result in results:
        camera_config = {
            "ip": result[2],
            "port": "554",  
            "username": result[0],
            "password": result[1],
            "camera": result[3],
            "channel": result[4]
        }
        cameras.append(camera_config)
    return cameras
    
def start_camera_process(branch, wait_insert, camera_config):
    """Start a camera process for a specific camera configuration."""
    camera_process = CameraProcess(branch, wait_insert, camera_config)
    camera_process.Person_Counter()

if __name__ == "__main__":
    branch_number = branch_value 
    wait_insert = 10
    cameras = fetch_all_cameras_for_branch(branch_number)
    threads = []
    for camera_config in cameras:
        thread = threading.Thread(target=start_camera_process, args=(branch_number, wait_insert, camera_config))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
