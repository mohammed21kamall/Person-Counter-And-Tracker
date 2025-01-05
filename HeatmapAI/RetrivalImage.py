import pymysql

class RetrieveImage:
    def __init__(self, Date: str):
        self._Date = Date

    def ConnectDB(self):
        return pymysql.connect(
            host='serastores.com',
            user='flow_admin',
            password='D4ta$Flow&2024',
            database='FlowAnalyticsDB'
        )

    def sanitize_filename(self, filename: str) -> str:
        # Replace invalid characters with underscores
        return filename.replace(":", "-").replace(" ", "_")

    def retrieve_images_from_database(self):
        try:
            connection = self.ConnectDB()
            cursor = connection.cursor()

            # Extract just the date part (YYYY-MM-DD) from the input date string
            date_str = self._Date.split(" ")[0]

            # Use DATE() to ignore the time component
            query = "SELECT id, image, capture_time FROM captured_images WHERE DATE(capture_time) = %s"
            cursor.execute(query, (date_str,))
            records = cursor.fetchall()

            if not records:
                print("No images found for the specified date.")
                return None

            output_paths = []  # List to store paths of retrieved images

            for record in records:
                image_id, image_data, capture_time = record

                # Sanitize the capture_time for use in the file name
                sanitized_time = self.sanitize_filename(str(capture_time))

                # Save the image
                output_path = f"captured_images/retrieved_image_{sanitized_time}.jpg"
                with open(output_path, 'wb') as file:
                    file.write(image_data)
                print(f"Retrieved image ID {image_id} captured at {capture_time}. Saved to {output_path}.")
                output_paths.append(output_path)

            return output_path  # Return list of output paths
        except pymysql.Error as e:
            print(f"Error while retrieving images from database: {e}")
        finally:
            if connection:
                cursor.close()
                connection.close()
                print("Database connection closed.")
