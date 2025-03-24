from scipy.ndimage import gaussian_filter
from RetrivalImage import RetrieveImage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymysql
import cv2
import os


class Heatmap:
    def __init__(self, date, branch, camera):
        self._date = date
        self._branch = branch
        self._camera = camera
        self._retrieval = RetrieveImage(self._branch, self._camera)
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
            date_str = self._date.split(" ")[0]
            branch = self._branch
            camera = self._camera

            movement_query = "SELECT X, Y FROM DirectionTest WHERE Branch = %s AND Camera = %s AND DATE(DateTime) = %s"
            cursor.execute(movement_query, (branch, camera, date_str))
            rows = cursor.fetchall()

            unique_person_query = "SELECT COUNT(DISTINCT PersonID) FROM DirectionTest WHERE Branch = %s AND Camera = %s AND DATE(DateTime) = %s"
            cursor.execute(unique_person_query, (branch, camera, date_str))
            unique_person_count = cursor.fetchone()[0]

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

        # Create figure with two subplots - one for heatmap, one for colorbar
        fig = plt.figure(figsize=(width / 30 + 2, height / 30))  # Added width for colorbar

        # Create main subplot for the heatmap
        ax_heatmap = plt.subplot(111)

        # Create and process heatmap data
        heatmap, _, _ = np.histogram2d(
            df['X'], df['Y'], bins=100, range=[[0, width], [0, height]]
        )
        heatmap = gaussian_filter(heatmap, sigma=2)
        heatmap = heatmap / np.max(heatmap)

        # Display base image
        ax_heatmap.imshow(frame)

        # Create heatmap overlay
        im = ax_heatmap.imshow(
            heatmap.T,
            cmap='jet',
            alpha=0.5,
            extent=[0, width, height, 0]
        )

        # Remove axes from main plot
        ax_heatmap.axis('off')

        # Add colorbar
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Value', size=10)

        # Set colorbar ticks
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0', '2', '4', '6', '8', '10'])

        # Add unique person count as text
        plt.text(
            0.05, 0.05, f"Unique Persons: {unique_person_count}",
            transform=ax_heatmap.transAxes,
            fontsize=25, color='white', weight='bold',
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5')
        )

        plt.tight_layout()

        filename_date = self._date.replace(' ', '_').replace(':', '-')
        output_file = os.path.join(f"Heatmap/Branch {self._branch}/Camera {self._camera}", f"Heatmap_{filename_date}.png")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        os.makedirs("Heatmap", exist_ok=True)

        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
