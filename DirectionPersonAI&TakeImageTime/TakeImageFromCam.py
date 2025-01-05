import pymysql
from datetime import datetime, timedelta
import requests
from requests.auth import HTTPDigestAuth
import urllib3
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CaptureCam:
    def __init__(self, Branch, CAPTURE_HOUR, CAPTURE_MINUTE, CAPTURE_SECOND):
        self._Branch = Branch
        self._cam = self.fetch_camera_config()
        self._CAPTURE_HOUR = CAPTURE_HOUR
        self._CAPTURE_MINUTE = CAPTURE_MINUTE
        self._CAPTURE_SECOND = CAPTURE_SECOND

        self.run_daily_capture(self._CAPTURE_HOUR, self._CAPTURE_MINUTE, self._CAPTURE_SECOND)

    def ConnectDB(self):
        """Establish a connection to the database."""
        return pymysql.connect(
            host="your_host.com",
            database="your_DB",
            user="your_user",
            password="your_password"
        )

    def fetch_camera_config(self):
        select_query = f"SELECT UserName, Password, IP FROM Camera WHERE Branch = {self._Branch}"
        connection = self.ConnectDB()
        cursor = connection.cursor()

        try:
            cursor.execute(select_query)
            result = cursor.fetchone()

            if result:
                return {
                    "ip": result[2],
                    "username": result[0],
                    "password": result[1],
                }
            else:
                print("No camera configuration found.")
                return None
        finally:
            cursor.close()
            connection.close()

    def wait_until_target_time(self, target_datetime: datetime) -> None:
        while True:
            current_time = datetime.now()
            if current_time >= target_datetime:
                print(f"Target time reached: {current_time}")
                break

            time_diff = (target_datetime - current_time).total_seconds()
            if time_diff > 60:
                print(f"Waiting... {time_diff / 60:.1f} minutes remaining")
                time.sleep(60)
            else:
                print(f"Waiting... {time_diff:.1f} seconds remaining")
                time.sleep(1)

    def capture_and_save_image(self, target_datetime: datetime) -> bool:
        try:

            if target_datetime > datetime.now():
                print(f"Waiting for target time: {target_datetime}")
                self.wait_until_target_time(target_datetime)

            formatted_time = target_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
            picture_url = f"https://{self._cam['ip']}/ISAPI/Streaming/channels/101/picture"
            params = {'playbacktime': formatted_time}

            print("Capturing image...")
            response = requests.get(
                picture_url,
                params=params,
                auth=HTTPDigestAuth(self._cam["username"], self._cam["password"]),
                verify=False,
                stream=True,
                timeout=30
            )

            if response.status_code == 200:
                image_data = response.content
                return self.insert_image_to_database(image_data, target_datetime)
            else:
                print(f"Failed to capture image. Status code: {response.status_code}")
                print(f"Response text: {response.text}")
                return False

        except Exception as e:
            print(f"Error capturing image: {str(e)}")
            return False

    def insert_image_to_database(self, image_data: bytes, capture_time: datetime) -> bool:
        connection = None
        try:
            connection = self.ConnectDB()
            cursor = connection.cursor()

            query = """
            INSERT INTO captured_images (image, capture_time)
            VALUES (%s, %s)
            """
            cursor.execute(query, (image_data, capture_time))
            connection.commit()
            print("Image successfully inserted into the database.")
            return True

        except pymysql.Error as e:
            print(f"Error while inserting image into database: {e}")
            return False
        finally:
            if connection:
                connection.close()
                print("Database connection closed.")

    def get_next_target_time(self, hour: int, minute: int, second: int = 0) -> datetime:
        """Calculate the next target time."""
        now = datetime.now()
        target_time = now.replace(hour=hour, minute=minute, second=second, microsecond=0)

        if target_time <= now:
            target_time += timedelta(days=1)

        return target_time

    def run_daily_capture(self, hour: int, minute: int, second: int = 0):
        """Run continuous daily capture at specified time."""
        print(f"Starting daily capture schedule at {hour:02d}:{minute:02d}:{second:02d}")

        while True:
            try:
                target_time = self.get_next_target_time(hour, minute, second)
                print(f"\nNext capture scheduled for: {target_time}")

                self.wait_until_target_time(target_time)

                success = self.capture_and_save_image(target_time)

                if success:
                    print("Daily capture completed successfully")
                else:
                    print("Daily capture failed")

                time.sleep(1)

            except Exception as e:
                print(f"Error in daily capture cycle: {str(e)}")
                print("Retrying in 60 seconds...")
                time.sleep(60)
