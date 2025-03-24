import pymysql
import cv2
import os

class RetrieveImage:
    def __init__(self, branch: int, camera: int):
        self._branch = branch
        self._camera = camera

    def ConnectDB(self):
        return pymysql.connect(
            host='serastores.com',
            user='flow_admin',
            password='D4ta$Flow&2024',
            database='FlowAnalyticsDB'
        )


    def resize_image(self, image_path: str, scale_factor: int = 500) -> str:
        # Read the image using OpenCV
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Image at {image_path} could not be read.")
            return image_path  # Return original path if not readable

        # Resize the image while maintaining aspect ratio
        new_width = scale_factor
        new_height = int(frame.shape[0] * (scale_factor / frame.shape[1]))
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Save the resized image
        resized_image_path = image_path.replace("retrieved_image_", "resized_image_")
        cv2.imwrite(resized_image_path, resized_frame)
        return resized_image_path

    def retrieve_images_from_database(self):
        connection = None
        try:
            connection = self.ConnectDB()
            cursor = connection.cursor()

            branch = self._branch
            camera = self._camera

            query = "SELECT id, image FROM captured_images WHERE Branch = %s AND Camera = %s"
            cursor.execute(query, (branch, camera))
            records = cursor.fetchall()

            if not records:
                print("No images found ...")
                return None


            for record in records:
                image_id, image_data = record

                # Sanitize the capture_time for use in the file name

                # Save the original image
                output_path = f"captured_images/Branch {self._branch}/Camera {self._camera}.jpg"

                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save the original image
                with open(output_path, 'wb') as file:
                    file.write(image_data)
                # Resize the saved image and get new path
                resized_image_path = self.resize_image(output_path)

            return resized_image_path  # Return list of output paths
        except pymysql.Error as e:
            print(f"Error while retrieving images from database: {e}")
        finally:
            if connection:
                cursor.close()
                connection.close()
                print("Database connection closed.")
