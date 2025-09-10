# Healthcare Monitoring App

## Overview
This is a Flask-based web application designed to process video input and estimate physiological parameters such as **heart rate**, **body temperature**, and **emotion** in real-time. The app uses computer vision and signal processing techniques to analyze facial videos, leveraging libraries like OpenCV, MediaPipe, SciPy, and DeepFace.

### Features
- **Heart Rate Estimation**: Extracts RGB signals from facial regions to compute heart rate using the chrominance method and a Kalman filter for smoothing.
- **Body Temperature Estimation**: Approximates body temperature based on RGB signal values.
- **Emotion Detection**: Analyzes facial expressions to detect emotions using the DeepFace library.
- **Real-Time Video Processing**: Processes uploaded video files and streams processed frames with overlaid heart rate and emotion information.
- **Server-Sent Events (SSE)**: Streams RGB and physiological data to the client for real-time monitoring.

## Prerequisites
- Python 3.8+
- A webcam or video file for input
- Dependencies listed in `requirements.txt` (see Installation section)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/stevesailab/contactless-healthcare-monitoring-.git
   cd healthcare-monitoring-app
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **You need to update the pip version first**:
```bash 
    python.exe -m pip install --upgrade pip
```


4. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
healthcare-monitoring-app/
├── app.py                  # Main Flask application
├── routes/
│   └── video_routes.py     # Routes for video processing and streaming
├── services/
│   └── video_processing.py # Core video processing logic
├── templates/
│   └── index.html          # Frontend HTML template (not provided, needs creation)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Usage
1. **Run the Application**:
   ```bash
   python app.py
   ```
   The app will start a Flask server on `http://localhost:5001`.

2. **Access the Web Interface**:
   - Open a browser and navigate to `http://localhost:5001`.
   - Upload a video file through the interface (you need to create an `index.html` with a file upload form).
   - The app will process the video and stream the output with heart rate and emotion overlays.

3. **API Endpoints**:
   - **POST /set_variable**: Resets Kalman filter and RGB buffers.
     ```json
     POST /set_variable
     Content-Type: application/json
     {"variable": "reset"}
     ```
   - **POST /upload**: Uploads a video file for processing.
     ```bash
     curl -X POST -F "video=@path/to/video.mp4" http://localhost:5001/upload
     ```
   - **GET /video_feed**: Streams processed video frames with overlays.
     ```bash
     curl http://localhost:5001/video_feed
     ```
   - **GET /rgb_stream**: Streams RGB and physiological data via SSE.
     ```bash
     curl http://localhost:5001/rgb_stream
     ```
   - **GET /get_rgb_data**: Retrieves the latest RGB and physiological data.
     ```bash
     curl http://localhost:5001/get_rgb_data
     ```

4. **Real-Time Monitoring**:
   - The app processes video frames to extract RGB signals from facial regions.
   - Heart rate is computed using a chrominance-based method and filtered with a bandpass filter (0.6–3.0 Hz) and Kalman smoothing.
   - Body temperature is estimated using an empirical formula based on RGB values.
   - Emotions are detected using DeepFace in a separate thread to avoid performance bottlenecks.
   - Processed frames are streamed with heart rate (or "Detecting") and emotion text overlays.

## Technical Details
- **Video Processing** (`video_processing.py`):
  - Uses MediaPipe Face Mesh to detect and mask facial regions.
  - Applies skin segmentation in HSV color space to isolate skin pixels.
  - Computes heart rate from RGB signals using the chrominance method, bandpass filtering, and FFT-based peak detection.
  - Estimates body temperature using an empirical formula: `36 + 0.01*R - 0.005*G + 0.008*B`.
  - Detects emotions using DeepFace in a separate thread for efficiency.
  - Streams processed frames as JPEG images with MJPEG format.

- **Flask Backend** (`app.py`, `video_routes.py`):
  - Handles video uploads and stores them in a temporary directory.
  - Provides endpoints for video streaming, real-time RGB data streaming via SSE, and fetching the latest physiological data.
  - Manages client connections for SSE streams and cleans up disconnected clients.

- **Threading**:
  - Emotion detection runs in a separate thread to prevent blocking the main video processing loop.
  - Thread-safe data access using locks for RGB data and emotion updates.

## Limitations
- **Accuracy**: Heart rate and temperature estimations are approximations and not medical-grade. They depend on lighting, video quality, and face detection accuracy.
- **Emotion Detection**: DeepFace may fail under poor lighting or with non-frontal faces, returning "Unknown" in such cases.
- **Performance**: Real-time processing is CPU-intensive, especially with DeepFace. The app uses a separate thread for emotion detection to mitigate this.
- **Frontend**: The provided code assumes an `index.html` template, which is not included. You need to create a frontend for video upload and display.

## Future Improvements
- Add a proper frontend with video upload, live streaming, and data visualization.
- Improve heart rate accuracy with advanced signal processing techniques.
- Optimize emotion detection for better performance and robustness.
- Add support for live webcam input in addition to video files.
- Include error handling for invalid video formats or corrupted files.

## Troubleshooting
- **No Face Detected**: Ensure the video contains a visible face and has good lighting.
- **Emotion Detection Fails**: Check DeepFace installation and model files. Set `enforce_detection=False` to handle cases with no face.
- **High CPU Usage**: Reduce the frequency of emotion detection (currently every 5th frame) or optimize DeepFace settings.
- **Port Conflicts**: Change the port in `app.py` if `5001` is in use.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (create one if needed).
