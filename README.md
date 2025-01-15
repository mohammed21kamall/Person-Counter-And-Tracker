# Person Counter, Direction Detection, and Heatmap Generation

## Overview

This project is designed to analyze video streams in real-time, count people, detect their movement directions, and generate heatmaps to visualize traffic patterns. It leverages computer vision techniques, including centroid tracking and object detection, to track people in a frame and log data for further analysis. The project is highly modular, allowing integration with various video feeds, including live RTSP streams, pre-recorded video files, and camera feeds from Hikvision cameras.

---

## Key Features

- **Real-Time People Counting**: Tracks the number of people entering and exiting a predefined zone.
- **Direction Detection**: Identifies whether individuals are moving upward or downward within the frame.
- **Heatmap Generation**: Creates visual representations of areas with high traffic intensity.
- **Database Logging**: Logs people count and timestamps into a MySQL database for historical analysis.
- **Camera Integration**: Supports RTSP streams and specific integration with Hikvision cameras.
- **Lightweight Executable**: Converts the code to an executable file for ease of deployment.

---

## Project Architecture

### Components

1. **CentroidTracker**:

   - Keeps track of detected objects using their centroids.
   - Assigns unique IDs to objects for tracking across frames.

2. **TrackableObject**:

   - Records movement history and calculates the direction of movement.
   - Provides state management for each tracked object.

3. **Direction Detection**:

   - Monitors the path of movement and determines whether the person is moving `up` or `down`.
   - Adjusts thresholds dynamically to account for varying video resolutions.

4. **Heatmap Module**:

   - Records movement paths and overlays them on a grid-based heatmap.
   - Generates and saves heatmaps for analysis.

5. **Database Integration**:

   - Connects to a MySQL database to store timestamps and people count.
   - Uses SQL queries to log and retrieve data efficiently.

6. **Hikvision Camera Integration**:

   - Configures RTSP streams using camera credentials.
   - Supports live streaming with optimized frame processing.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- MySQL Server
- Required Python Libraries (listed in `requirements.txt`)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/mohammed21kamall/person_counter_direction_heatmap.git
   cd Person_Counter_And_Tracker
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the MySQL database:

   - Create a database and table using the provided `schema.sql` file.
   - Update the database credentials in the `config.py` file.

4. Configure the camera:

   - Update the `camera_config.py` file with the IP address, username, password, and RTSP URL of your camera.

5. Run the application:

   ```bash
   python CameraAI.py
   ```

---

## How It Works

### Real-Time Processing

1. **Frame Capture**:

   - Captures frames from a video source (RTSP, file, or camera).
   - Resizes frames for consistent processing speed.

2. **Object Detection**:

   - Detects people in the frame using a pre-trained YOLO or SSD model.
   - Outputs bounding boxes for detected individuals.

3. **Tracking**:

   - Assigns a unique ID to each detected person.
   - Tracks movement across consecutive frames using centroid tracking.

4. **Direction Detection**:

   - Analyzes the movement trajectory of each tracked object.
   - Updates counts for upward and downward movements.

5. **Database Logging**:

   - Records the current timestamp and counts in the database.
   - Ensures accurate logging with minimal latency.

### Heatmap Generation

1. **Data Collection**:

   - Records the coordinates of all detected objects over time.

2. **Visualization**:

   - Aggregates coordinates into a 2D grid.
   - Generates heatmaps showing areas of high activity.

3. **Output**:

   - Saves heatmaps as images or overlays them on video frames.

---

## Configuration

### Configuration Files

1. **`config.py`**:

   - Stores general application settings, including database credentials and logging options.

2. **`camera_config.py`**:

   - Contains camera-specific settings such as RTSP URLs and authentication credentials.

3. **`CameraAI.py`**:

   - Entry point of the application, tying together all modules.

### Modifiable Parameters

- **Frame Skip Rate**: Adjust the number of frames to skip for faster processing (and skipped 2 fps).
- **Heatmap Resolution**: Change the granularity of heatmaps by modifying grid size.
- **Database Intervals**: Set the frequency of logging entries in seconds.

---

## Usage Examples

### Example 1: People Counting and Direction Detection  
Use the following command to process a live RTSP stream:  
`python main.py --rtsp rtsp://username:password@camera_ip/stream/channel`  
The application will display real-time counts and directions on the video feed.  

<video width="640" height="360" controls>
  <source src="[PersonCounterVideo/video.mp4](https://github.com/mohammed21kamall/Person-Counter-And-Tracker/blob/main/PersonCounterVideo/video.mp4)" type="video/mp4">
  Your browser does not support the video tag.
</video>



### Example 2: Generate Heatmap from images

-The Heatmap code and its dedicated GUI retrieve all the movements that the Direction code has confined to my database by entering the date of the day for which I want to know the heatmap and also retrieving the image that was taken from the readable camera and doing the Heatmap after that... Here is a picture to explain the Heatmap :

- **Before Heatmap**  
  ![Before Heatmap](captured_images/retrieved_image_2024-12-31_23-59-59.jpg)


- **After Heatmap**  
  ![After Heatmap](HeatmapAI/HeatmapImages/Heatmap_2024-12-31.png)


### Example 3: Database Logging

- Ensure the MySQL server is running and the database is configured.
- Run the application to log counts at regular intervals.

---

## Deployment

### Creating an Executable

To convert the application to an executable:

1. Install `auto-py-to-exe`:
   ```bash
   pip install auto-py-to-exe
   ```
2. Launch the GUI:
   ```bash
   auto-py-to-exe
   ```
3. Configure the script and output settings in the GUI.
4. Build the executable and distribute it as needed.

### Optimizing the Executable Size

- Use the `--exclude` option to remove unnecessary packages.
- Minify the libraries by bundling only required dependencies.

---

## Troubleshooting

### Common Issues

1. **RTSP Stream Not Working**:

   - Ensure the camera IP and credentials are correct.
   - Test the RTSP URL using a media player like VLC.

2. **Database Connection Errors**:

   - Verify the MySQL server is running.
   - Check the database credentials in `config.py`.

3. **Heatmap Not Generated**:

   - Confirm that movement data is being recorded.
   - Check the output directory for saved heatmap images.

### Logs and Debugging

- Enable verbose logging by setting `DEBUG = True` in `config.py`.
- Check the `logs/` directory for detailed error messages.

---

## Future Enhancements

- **Multi-Camera Support**: Extend the system to handle multiple simultaneous streams.
- **Advanced Analytics**: Integrate AI-based insights for traffic prediction.
- **Cloud Integration**: Store heatmaps and logs in the cloud for remote access.
- **Crossing Zone Detection**: Add support for customizable crossing zones.

---

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure your code adheres to the PEP 8 standard and includes relevant documentation.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or support, contact Mohammed at `mk8492977@gmail.com`.

