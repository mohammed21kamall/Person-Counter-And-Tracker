import os
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
from RetrivalImage import RetrieveImage


class Heatmap:
    def __init__(self, date):
        self._date = date
        self._retrieval = RetrieveImage(self._date)
        self._frame_image = self._retrieval.retrieve_images_from_database()

    def connect_to_db(self):
        try:
            return pymysql.connect(
                host=os.getenv('DB_HOST', "serastores.com"),
                database=os.getenv('DB_NAME', "FlowAnalyticsDB"),
                user=os.getenv('DB_USER', "flow_admin"),
                password=os.getenv('DB_PASSWORD', "D4ta$Flow&2024")
            )
        except pymysql.MySQLError as e:
            print(f"Database connection failed: {e}")
            return None

    def create_heatmap_overlay(self):
        conn = self.connect_to_db()
        if not conn:
            print("Failed to connect to the database.")
            return

        try:
            cursor = conn.cursor()
            # Extract the date part from the input date string for comparison
            date_str = self._date.split(" ")[0]  # Get just the date (YYYY-MM-DD)
            query = f"SELECT X, Y FROM PersonMovement WHERE DATE(DateTime) = '{date_str}'"
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
        except pymysql.MySQLError as e:
            print(f"Database query failed: {e}")
            return

        if not rows:
            print("No movement data retrieved.")
            return

        df = pd.DataFrame(rows, columns=['X', 'Y'])

        frame = cv2.imread(self._frame_image)
        if frame is None:
            print(f"Failed to load frame image: {self._frame_image}")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        plt.figure(figsize=(width / 100, height / 100))
        heatmap, _, _ = np.histogram2d(
            df['X'], df['Y'], bins=100, range=[[0, width], [0, height]]
        )
        heatmap = gaussian_filter(heatmap, sigma=2)
        heatmap = heatmap / np.max(heatmap)

        plt.imshow(frame)
        plt.imshow(
            heatmap.T, cmap='jet', alpha=0.6, extent=[0, width, height, 0]
        )
        plt.axis('off')

        # Format date string for filename
        filename_date = self._date.replace(' ', '_').replace(':', '-')
        output_file = os.path.join("Heatmap", f"Heatmap_{filename_date}.png")

        # Ensure output directory exists
        os.makedirs("Heatmap", exist_ok=True)

        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Heatmap saved to {output_file}")



