import os
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
from RetrivalImage import RetrieveImage

class Heatmap:
    def __init__(self, date, branch, camera):
        self._date = date
        self._branch = branch
        self._camera = camera
        self._retrieval = RetrieveImage(self._date, self._branch, self._camera)
        self._frame_image = self._retrieval.retrieve_images_from_database()
        
    def connect_to_db(self):
        try:
            return pymysql.connect(
                host=os.getenv('DB_HOST', "host.com"),
                database=os.getenv('DB_NAME', "NameDB"),
                user=os.getenv('DB_USER', "your_user"),
                password=os.getenv('DB_PASSWORD', "PasswordDB")
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
        fig = plt.figure(figsize=(width / 30 + 2, height / 30)) 
        ax_heatmap = plt.subplot(111)
        heatmap, _, _ = np.histogram2d(
            df['X'], df['Y'], bins=100, range=[[0, width], [0, height]]
        )
        heatmap = gaussian_filter(heatmap, sigma=2)
        heatmap = heatmap / np.max(heatmap)
        ax_heatmap.imshow(frame)
        im = ax_heatmap.imshow(
            heatmap.T,
            cmap='jet',
            alpha=0.6,
            extent=[0, width, height, 0]
        )
        ax_heatmap.axis('off')
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7]) 
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Value', size=10)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0', '2', '4', '6', '8', '10'])
        plt.text(
            0.05, 0.05, f"Unique Persons: {unique_person_count}",
            transform=ax_heatmap.transAxes,
            fontsize=25, color='white', weight='bold',
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5')
        )
        plt.tight_layout()
        filename_date = self._date.replace(' ', '_').replace(':', '-')
        output_file = os.path.join("Heatmap", f"Heatmap_{filename_date}.png")
        os.makedirs("Heatmap", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Heatmap saved to {output_file}")
        print(f"Number of unique persons: {unique_person_count}")
