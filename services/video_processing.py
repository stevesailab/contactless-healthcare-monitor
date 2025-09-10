# import cv2
# import numpy as np
# import mediapipe as mp
# from scipy.signal import butter, filtfilt
# from collections import deque
# import threading
# import time
# import queue
# from deepface import DeepFace

# BUFFER_SIZE = 300
# FPS = 30
# rgb_buffer = {'r': deque(maxlen=BUFFER_SIZE), 'g': deque(maxlen=BUFFER_SIZE), 'b': deque(maxlen=BUFFER_SIZE)}
# hr_values = deque(maxlen=10)
# rgb_clients = []
# current_rgb_data = {'r': 0, 'g': 0, 'b': 0, 'heart_rate': 75.0}
# rgb_data_lock = threading.Lock()

# kalman_hr = 75.0
# kalman_p = 1.0
# kalman_q = 0.01
# kalman_r = 1.0

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# def set_kalman_defaults():
#     global kalman_hr, kalman_p
#     rgb_buffer['r'].clear()
#     rgb_buffer['g'].clear()
#     rgb_buffer['b'].clear()
#     hr_values.clear()
#     kalman_hr = 75.0
#     kalman_p = 1.0

# def blackout_outside_dynamic_threshold(frame, lower_factor=0.48, upper_factor=1.74):
#     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     non_zero_pixels = gray_image[gray_image != 0]
#     average_value = np.median(non_zero_pixels)
#     lower_threshold = max(0, int(average_value * lower_factor))
#     upper_threshold = min(255, int(average_value * upper_factor))
#     mask = (gray_image >= lower_threshold) & (gray_image <= upper_threshold)
#     updated_frame = np.zeros_like(frame)
#     updated_frame[mask] = frame[mask]
#     return updated_frame

# def bandpass_filter(signal, lowcut=0.6, highcut=3.0, fs=FPS, order=5):
#     nyquist = 0.5 * fs
#     low, high = lowcut / nyquist, highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return filtfilt(b, a, signal) if len(signal) > order else signal

# def chrom_method(r_signal, g_signal, b_signal):
#     X = 3 * np.array(r_signal) - 2 * np.array(g_signal)
#     Y = 1.5 * np.array(g_signal) - 1.5 * np.array(b_signal)
#     return X + Y

# def compute_heart_rate(signal):
#     global kalman_hr, kalman_p
#     if len(signal) < BUFFER_SIZE:
#         return kalman_hr
#     filtered_signal = bandpass_filter(signal)
#     fft_spectrum = np.fft.rfft(filtered_signal)
#     freqs = np.fft.rfftfreq(len(filtered_signal), d=1/FPS)
#     valid_range = (freqs >= 0.92) & (freqs <= 2.0)
#     if not any(valid_range):
#         return kalman_hr
#     peak_freq = freqs[valid_range][np.argmax(np.abs(fft_spectrum[valid_range]))]
#     bpm = peak_freq * 60
#     if bpm < 40:
#         bpm = kalman_hr
#     hr_values.append(bpm)
#     bpm_smoothed = np.mean(hr_values)
#     kalman_p += kalman_q
#     kalman_k = kalman_p / (kalman_p + kalman_r)
#     kalman_hr = kalman_hr + kalman_k * (bpm_smoothed - kalman_hr)
#     kalman_p = (1 - kalman_k) * kalman_p
#     return kalman_hr

# def extract_rgb_signals(frame):
#     face_pixels = frame[frame.sum(axis=2) > 0]
#     if len(face_pixels) == 0:
#         return (0, 0, 0)
#     mean_r = int(np.mean(face_pixels[:, 2]))
#     mean_g = int(np.mean(face_pixels[:, 1]))
#     mean_b = int(np.mean(face_pixels[:, 0]))
#     return (mean_r, mean_g, mean_b)

# def create_face_mask_with_colors(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_image)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             points = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
#                                for landmark in face_landmarks.landmark], dtype=np.int32)
#             hull = cv2.convexHull(points)
#             cv2.fillPoly(mask, [hull], 255)
#     masked_face = cv2.bitwise_and(image, image, mask=mask)
#     return masked_face

# def skin_segmentation(face_mask):
#     hsv = cv2.cvtColor(face_mask, cv2.COLOR_BGR2HSV)
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#     skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
#     return cv2.bitwise_and(face_mask, face_mask, mask=skin_mask)

# def broadcast_rgb_data(data):
#     with rgb_data_lock:
#         global current_rgb_data
#         current_rgb_data = data
#         disconnected_clients = []
#         for client_queue in rgb_clients:
#             try:
#                 client_queue.put(data, timeout=0.1)
#             except queue.Full:
#                 disconnected_clients.append(client_queue)
#         for client in disconnected_clients:
#             rgb_clients.remove(client)
# # In video_processing.py
# def generate_gray_frames(path):
#     cap = cv2.VideoCapture(path)
#     frame_count = 0
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#         frame_count += 1
#         face_mask = create_face_mask_with_colors(frame)
#         if face_mask is None or not np.any(face_mask):
#             cv2.putText(frame, "No face detected", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#             rgb_data = {'r': 0, 'g': 0, 'b': 0, 'heart_rate': 0.0}
#         else:
#             skin_segmented = skin_segmentation(face_mask)
#             face = blackout_outside_dynamic_threshold(skin_segmented)
#             mean_r, mean_g, mean_b = extract_rgb_signals(face)
#             rgb_buffer["r"].append(mean_r)
#             rgb_buffer["g"].append(mean_g)
#             rgb_buffer["b"].append(mean_b)
#             pulse_signal = chrom_method(rgb_buffer['r'], rgb_buffer['g'], rgb_buffer['b'])
#             heart_rate = compute_heart_rate(pulse_signal)
#             if heart_rate == 75:
#                 heart_rate = "Detecting"
#             estimated_temp = 36 + 0.01 * mean_r - 0.005 * mean_g + 0.008 * mean_b
            
#             # Detect emotion for the current frame
#             emotion = detect_emotion(frame)
            
#             # Overlay emotion text on the frame
#             cv2.putText(frame, f'Emotion: {emotion}', (10, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

#             rgb_data = {
#                 'r': mean_r,
#                 'g': mean_g,
#                 'b': mean_b,
#                 'heart_rate': heart_rate,
#                 'timestamp': time.time(),
#                 'body_temprature': estimated_temp,
#                 'emotion': emotion  # Add emotion to rgb_data for potential use
#             }
#         if frame_count % 3 == 0:
#             broadcast_rgb_data(rgb_data)
#         ret, buffer = cv2.imencode('.jpg', frame)  # Encode the frame with emotion text
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     cap.release()

# # Function to detect emotion using DeepFace
# def detect_emotion(image):
   
#     try:
       
#         result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
#         emotion = result[0]['dominant_emotion']  # Get the dominant emotion
#         return emotion
#     except Exception as e:
#         print(f"Error detecting emotion: {e}")
#         return "Unknown"




# video_processing.py
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt
from collections import deque
import threading
import time
import queue
from deepface import DeepFace
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# BUFFER_SIZE = 300
BUFFER_SIZE = 60
FPS = 30
rgb_buffer = {'r': deque(maxlen=BUFFER_SIZE), 'g': deque(maxlen=BUFFER_SIZE), 'b': deque(maxlen=BUFFER_SIZE)}
hr_values = deque(maxlen=10)
rgb_clients = []
current_rgb_data = {'r': 0, 'g': 0, 'b': 0, 'heart_rate': 75.0, 'emotion': 'Unknown'}
rgb_data_lock = threading.Lock()

kalman_hr = 75.0
kalman_p = 1.0
kalman_q = 0.01
kalman_r = 1.0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

latest_emotion = 'Unknown'
emotion_lock = threading.Lock()

def set_kalman_defaults():
    global kalman_hr, kalman_p
    rgb_buffer['r'].clear()
    rgb_buffer['g'].clear()
    rgb_buffer['b'].clear()
    hr_values.clear()
    kalman_hr = 75.0
    kalman_p = 1.0
    with emotion_lock:
        global latest_emotion
        latest_emotion = 'Unknown'

def blackout_outside_dynamic_threshold(frame, lower_factor=0.48, upper_factor=1.74):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    non_zero_pixels = gray_image[gray_image != 0]
    average_value = np.median(non_zero_pixels) if non_zero_pixels.size > 0 else 0
    lower_threshold = max(0, int(average_value * lower_factor))
    upper_threshold = min(255, int(average_value * upper_factor))
    mask = (gray_image >= lower_threshold) & (gray_image <= upper_threshold)
    updated_frame = np.zeros_like(frame)
    updated_frame[mask] = frame[mask]
    return updated_frame

def bandpass_filter(signal, lowcut=0.6, highcut=3.0, fs=FPS, order=5):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal) if len(signal) > order else signal

def chrom_method(r_signal, g_signal, b_signal):
    X = 3 * np.array(r_signal) - 2 * np.array(g_signal)
    Y = 1.5 * np.array(g_signal) - 1.5 * np.array(b_signal)
    return X + Y

def compute_heart_rate(signal):
    global kalman_hr, kalman_p
    if len(signal) < BUFFER_SIZE:
        logger.debug(f"Buffer not full: {len(signal)}/{BUFFER_SIZE}, returning default HR: {kalman_hr}")
        return kalman_hr
    filtered_signal = bandpass_filter(signal)
    fft_spectrum = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(len(filtered_signal), d=1/FPS)
    valid_range = (freqs >= 0.92) & (freqs <= 2.0)
    if not any(valid_range):
        logger.debug(f"No valid frequencies in range, returning current HR: {kalman_hr}")
        return kalman_hr
    peak_freq = freqs[valid_range][np.argmax(np.abs(fft_spectrum[valid_range]))]
    bpm = peak_freq * 60
    logger.debug(f"Computed BPM: {bpm}")
    if bpm < 40:
        logger.debug(f"BPM {bpm} too low, returning current HR: {kalman_hr}")
        bpm = kalman_hr
    hr_values.append(bpm)
    bpm_smoothed = np.mean(hr_values)
    kalman_p += kalman_q
    kalman_k = kalman_p / (kalman_p + kalman_r)
    kalman_hr = kalman_hr + kalman_k * (bpm_smoothed - kalman_hr)
    kalman_p = (1 - kalman_k) * kalman_p
    logger.debug(f"Smoothed HR: {kalman_hr}")
    return kalman_hr


def compute_heart_rate(signal):
    global kalman_hr, kalman_p
    if len(signal) < BUFFER_SIZE:
        return kalman_hr
    filtered_signal = bandpass_filter(signal)
    fft_spectrum = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(len(filtered_signal), d=1/FPS)
    valid_range = (freqs >= 0.92) & (freqs <= 2.0)
    if not any(valid_range):
        return kalman_hr
    peak_freq = freqs[valid_range][np.argmax(np.abs(fft_spectrum[valid_range]))]
    bpm = peak_freq * 60
    if bpm < 40:
        bpm = kalman_hr
    hr_values.append(bpm)
    bpm_smoothed = np.mean(hr_values)
    kalman_p += kalman_q
    kalman_k = kalman_p / (kalman_p + kalman_r)
    kalman_hr = kalman_hr + kalman_k * (bpm_smoothed - kalman_hr)
    kalman_p = (1 - kalman_k) * kalman_p
    return kalman_hr
def extract_rgb_signals(frame):
    face_pixels = frame[frame.sum(axis=2) > 0]
    if len(face_pixels) == 0:
        logger.debug("No face pixels detected")
        return (0, 0, 0)
    mean_r = int(np.mean(face_pixels[:, 2]))
    mean_g = int(np.mean(face_pixels[:, 1]))
    mean_b = int(np.mean(face_pixels[:, 0]))
    logger.debug(f"RGB signals: R={mean_r}, G={mean_g}, B={mean_b}")
    return (mean_r, mean_g, mean_b)

def create_face_mask_with_colors(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                               for landmark in face_landmarks.landmark], dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)
    masked_face = cv2.bitwise_and(image, image, mask=mask)
    return masked_face

def skin_segmentation(face_mask):
    hsv = cv2.cvtColor(face_mask, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return cv2.bitwise_and(face_mask, face_mask, mask=skin_mask)

def broadcast_rgb_data(data):
    with rgb_data_lock:
        global current_rgb_data
        current_rgb_data = data
        logger.debug(f"Broadcasting RGB data: {data}")
        disconnected_clients = []
        for client_queue in rgb_clients:
            try:
                client_queue.put(data, timeout=0.1)
            except queue.Full:
                disconnected_clients.append(client_queue)
        for client in disconnected_clients:
            rgb_clients.remove(client)

def detect_emotion_thread(frame_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
            emotion = detect_emotion(frame)
            with emotion_lock:
                global latest_emotion
                latest_emotion = emotion
            logger.debug(f"Detected emotion: {emotion}")
            frame_queue.task_done()
        except queue.Empty:
            continue

def generate_gray_frames(path):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    emotion_thread = threading.Thread(target=detect_emotion_thread, args=(frame_queue, stop_event))
    emotion_thread.daemon = True
    emotion_thread.start()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.debug("Failed to read frame or end of video")
                break
            frame_count += 1
            face_mask = create_face_mask_with_colors(frame)
            if face_mask is None or not np.any(face_mask):
                logger.debug("No face detected in frame")
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                rgb_data = {'r': 0, 'g': 0, 'b': 0, 'heart_rate': 0.0, 'emotion': 'Unknown'}
            else:
                skin_segmented = skin_segmentation(face_mask)
                face = blackout_outside_dynamic_threshold(skin_segmented)
                mean_r, mean_g, mean_b = extract_rgb_signals(face)
                rgb_buffer["r"].append(mean_r)
                rgb_buffer["g"].append(mean_g)
                rgb_buffer["b"].append(mean_b)
                pulse_signal = chrom_method(rgb_buffer['r'], rgb_buffer['g'], rgb_buffer['b'])
                heart_rate = compute_heart_rate(pulse_signal)
                if heart_rate == 75:
                    heart_rate = "Detecting"
                estimated_temp = 36 + 0.01 * mean_r - 0.005 * mean_g + 0.008 * mean_b

                if frame_count % 5 == 0:
                    try:
                        frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        logger.debug("Emotion queue full, skipping frame")
                        pass

                with emotion_lock:
                    emotion = latest_emotion

                # Overlay heart rate and emotion on the frame
                hr_display = heart_rate if isinstance(heart_rate, str) else f"{heart_rate:.1f} BPM"
                # cv2.putText(frame, f'Heart Rate: {hr_display}', (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Emotion: {emotion}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                rgb_data = {
                    'r': mean_r,
                    'g': mean_g,
                    'b': mean_b,
                    'heart_rate': heart_rate,
                    'timestamp': time.time(),
                    'body_temprature': estimated_temp,
                    'emotion': emotion
                }
            if frame_count % 3 == 0:
                broadcast_rgb_data(rgb_data)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        stop_event.set()
        cap.release()
        logger.debug("Video capture released")

def detect_emotion(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        logger.error(f"Error detecting emotion: {e}")
        return "Unknown"