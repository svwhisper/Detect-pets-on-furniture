import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import atexit
import json
import os
import ftplib
import paho.mqtt.client as mqtt
import sys
import datetime
import torch

# --- TIMESTAMP LOGGING CLASS ---
class TimestampWriter:
    def __init__(self, stream):
        self.stream = stream
        self.new_line = True
    def write(self, message):
        if not message: return
        parts = message.split('\n')
        for i, part in enumerate(parts):
            if self.new_line and part:
                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                self.stream.write(timestamp)
                self.new_line = False
            self.stream.write(part)
            if i < len(parts) - 1:
                self.stream.write('\n')
                self.new_line = True
    def flush(self):
        self.stream.flush()

sys.stdout = TimestampWriter(sys.stdout)
sys.stderr = TimestampWriter(sys.stderr)

# --- LOAD CONFIGURATION ---
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("CRITICAL ERROR: config.json not found!")
    sys.exit(1)
except json.JSONDecodeError:
    print("CRITICAL ERROR: config.json is invalid JSON!")
    sys.exit(1)

# Extract config vars
RTSP_URL = config.get("rtsp_url", "")
SNAPSHOT_DIR = config.get("snapshot_dir", "snapshots")
MQTT_BROKER = config["mqtt"]["broker"]
MQTT_PORT = config["mqtt"]["port"]
MQTT_TOPIC_STATUS = config["mqtt"]["topic_status"]
MQTT_TOPIC_ERROR = config["mqtt"]["topic_error"]
FTP_SERVER = config["ftp"]["server"]
FTP_USER = config["ftp"]["user"]
FTP_PASS = config["ftp"]["pass"]
FTP_DIR = config["ftp"]["dir"]
MODEL_NAME = config["model"]["name"]
CONFIDENCE_THRESHOLD = config["model"]["confidence_threshold"]
IOU_THRESHOLD = config["model"]["iou_threshold"]
FRAME_SKIP = config["tuning"]["frame_skip"]
CLEAR_THRESHOLD_CHECKS = config["tuning"]["clear_threshold_checks"]
ERROR_COOLDOWN = config["tuning"]["error_cooldown_seconds"]
CAMERA_ERROR_THRESHOLD = config["tuning"]["camera_error_threshold"]
EROSION_SIZE_SOFA = config["tuning"]["erosion_sofa"]
EROSION_SIZE_CHAIR = config["tuning"]["erosion_chair"]
MAX_MISSED_FRAMES = config["persistence"]["max_missed_frames"]

# Snapshot Cooldown
SNAPSHOT_COOLDOWN = config.get("tuning", {}).get("snapshot_cooldown", 300) 

def calculate_backoff(retry_count, max_delay=60):
    return min(max_delay, 2 ** retry_count)

# --- MODEL LOADING ---
coreml_path = f"{MODEL_NAME}.mlpackage"
pt_path = f"{MODEL_NAME}.pt"
use_coreml = False
print(f"Initializing {MODEL_NAME}...")
try:
    import coremltools
    has_coreml_tools = True
except ImportError:
    has_coreml_tools = False

can_write = os.access('.', os.W_OK)
if not can_write:
    print("WARNING: No write permission. Model export skipped.")

if has_coreml_tools and can_write:
    if not os.path.exists(coreml_path):
        print(f"CoreML model not found. Attempting export...")
        try:
            model_pt = YOLO(pt_path)
            model_pt.export(format="coreml", nms=False) 
            print("Export success.")
            use_coreml = True
        except Exception as e:
            print(f"ANE Export failed: {e}")
            use_coreml = False
    else:
        use_coreml = True
else:
    if torch.cuda.is_available():
        print("NVIDIA GPU detected.")
    else:
        print("Using CPU.")

if use_coreml and os.path.exists(coreml_path):
    print(f"Loading ANE Model: {coreml_path}")
    model = YOLO(coreml_path, task="segment")
else:
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Loading Standard Model ({device}): {pt_path}")
    model = YOLO(pt_path, task="segment")

if not os.path.exists(SNAPSHOT_DIR):
    try:
        os.makedirs(SNAPSHOT_DIR)
    except OSError:
        print(f"CRITICAL: Cannot create snapshot dir {SNAPSHOT_DIR}.")

# --- MQTT SETUP ---
client = mqtt.Client()
def on_connect(client, userdata, flags, rc):
    if rc == 0: print(f"Connected to MQTT Broker at {MQTT_BROKER}")
    else: print(f"MQTT Connection Failed with code {rc}")
client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
except Exception as e:
    print(f"Failed to initiate MQTT: {e}")

# --- STATE VARIABLES ---
current_state = "clear"
last_published_state = None
consecutive_miss_checks = 0
global_furniture_data = [] 
missed_furniture_frames = 0
last_error_msg = None
last_error_time = 0
last_camera_error_msg = None
last_camera_error_time = 0

# FIX: Missing Variable Restored
last_valid_animal = "clear"

# Track last snapshot time
last_snapshot_time = 0 

def publish_state(new_state, frame_image):
    global last_published_state, last_snapshot_time
    
    if new_state != last_published_state:
        if new_state == "clear":
            client.publish(MQTT_TOPIC_STATUS, new_state, retain=True)
            print(f"MQTT PUBLISH: {MQTT_TOPIC_STATUS} -> {new_state}")
        
        elif new_state in ["dog", "cat"]:
            current_time = time.time()
            if (current_time - last_snapshot_time) > SNAPSHOT_COOLDOWN:
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                filename = f"{SNAPSHOT_DIR}/{new_state}_{timestamp}.jpg"
                try:
                    cv.imwrite(filename, frame_image)
                    print(f"Snapshot saved: {filename}")
                    last_snapshot_time = current_time 
                except Exception as e:
                    print(f"Failed to write snapshot: {e}")
                    return
                
                try:
                    ftp = ftplib.FTP(FTP_SERVER, FTP_USER, FTP_PASS, timeout=10)
                    ftp.cwd(FTP_DIR)
                    with open(filename, 'rb') as file_obj:
                        ftp.storbinary(f"STOR {os.path.basename(filename)}", file_obj)
                    ftp.quit()
                except Exception as e:
                    print(f"FTP Transfer Failed: {e}")

                payload = f"{new_state} {os.path.basename(filename)}"
            else:
                payload = f"{new_state} (rate_limited)"
                print(f"Snapshot skipped (Cooldown active).")

            client.publish(MQTT_TOPIC_STATUS, payload, retain=True)
            print(f"MQTT PUBLISH: {MQTT_TOPIC_STATUS} -> {payload}")
        
        last_published_state = new_state

def publish_error(error_msg):
    global last_error_msg, last_error_time
    current_time = time.time()
    if error_msg != last_error_msg or (current_time - last_error_time > ERROR_COOLDOWN):
        client.publish(MQTT_TOPIC_ERROR, error_msg)
        print(f"MQTT ERROR PUBLISH: {error_msg}")
        last_error_msg = error_msg
        last_error_time = current_time

def publish_camera_failure(reason):
    global last_camera_error_msg, last_camera_error_time
    current_time = time.time()
    if reason != last_camera_error_msg or (current_time - last_camera_error_time > ERROR_COOLDOWN):
        error_payload = f"Error: {reason}"
        client.publish(MQTT_TOPIC_STATUS, error_payload, retain=True)
        print(f"MQTT CRITICAL FAIL: {MQTT_TOPIC_STATUS} -> {error_payload}")
        last_camera_error_msg = reason
        last_camera_error_time = current_time

def cleanup():
    client.loop_stop()
    client.disconnect()
    print("Exiting.")
atexit.register(cleanup)

# --- MAIN LOOP ---
if RTSP_URL == "":
    RTSP_URL = str(input("Please enter your RTSP URL: "))
safe_url_log = RTSP_URL.split('@')[-1] if '@' in RTSP_URL else "RTSP Stream"
print(f"Connecting to: {safe_url_log}")

video_feed = cv.VideoCapture(RTSP_URL)
if not video_feed.isOpened():
    print("Could not open video feed at startup.")
    publish_camera_failure("Could not open video feed at startup")
    video_feed.release()

frame_count = 0
consecutive_camera_errors = 0
retry_counter = 0

print(f"Starting Monitor. IOU={IOU_THRESHOLD}, Skip={FRAME_SKIP}, SnapshotCooldown={SNAPSHOT_COOLDOWN}s...")

while True:
    try:
        if not video_feed.isOpened():
            backoff_time = calculate_backoff(retry_counter)
            print(f"Attempting reconnection in {backoff_time}s...")
            time.sleep(backoff_time)
            video_feed = cv.VideoCapture(RTSP_URL)
            if not video_feed.isOpened():
                retry_counter += 1
                consecutive_camera_errors += 1
                continue
            else:
                print("Reconnection Successful.")
                retry_counter = 0 
                consecutive_camera_errors = 0

        ret, frame = video_feed.read()
        if not ret:
            consecutive_camera_errors += 1
            print(f"Frame read failed ({consecutive_camera_errors}/{CAMERA_ERROR_THRESHOLD})...")
            if consecutive_camera_errors >= CAMERA_ERROR_THRESHOLD:
                publish_camera_failure("Camera frame read failed")
                video_feed.release()
                retry_counter = 1
                continue
            time.sleep(0.1) 
            continue
            
        consecutive_camera_errors = 0
        retry_counter = 0 
        frame_count += 1
        if frame_count % (FRAME_SKIP + 1) != 0:
            continue

        frame = cv.resize(frame, (640, 480))
        if last_camera_error_msg is not None:
            last_camera_error_msg = None

        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        result = results[0]

        current_furniture_data = [] 
        pet_contours = [] 
        person_present = False

        if result.masks is not None:
            masks_points = result.masks.xy 
            classes = result.boxes.cls.cpu().numpy()
            
            for i, class_id in enumerate(classes):
                if i >= len(masks_points): continue
                points = masks_points[i]
                if len(points) == 0: continue
                
                contour = np.array(points, dtype=np.int32)
                cls_int = int(class_id)
                
                if cls_int in [56, 57, 59]: 
                    area = cv.contourArea(contour)
                    if area > (307200 * 0.60): 
                        continue
                    current_furniture_data.append((contour, cls_int))
                    cv.polylines(frame, [contour], True, (255, 200, 0), 2)
                elif cls_int == 0: 
                    person_present = True
                    cv.polylines(frame, [contour], True, (0, 0, 255), 2)
                elif cls_int in [15, 16]: 
                    pet_name = "dog" if cls_int == 16 else "cat"
                    pet_contours.append((pet_name, contour))
                    cv.polylines(frame, [contour], True, (0, 255, 0), 2)

        # --- MEMORY LOGIC ---
        if current_furniture_data:
            global_furniture_data = current_furniture_data
            missed_furniture_frames = 0
        else:
            missed_furniture_frames += 1
            if missed_furniture_frames > MAX_MISSED_FRAMES:
                global_furniture_data = [] 

        furniture_to_use = []
        if global_furniture_data:
            furniture_to_use = global_furniture_data
            if not current_furniture_data:
                 for cnt, _ in global_furniture_data:
                     cv.polylines(frame, [cnt], True, (100, 100, 0), 1)
                 cv.putText(frame, "(Memory)", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 0), 1)

        dog_detected_on_couch = False
        cat_detected_on_couch = False

        if not furniture_to_use:
            cv.putText(frame, "NO FURNITURE FOUND", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            total_furniture_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for cnt, cls_id in furniture_to_use:
                item_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv.fillPoly(item_mask, [cnt], 255)
                
                if cls_id == 56: e_size = EROSION_SIZE_CHAIR
                else: e_size = EROSION_SIZE_SOFA
                
                kernel = np.ones((e_size, e_size), np.uint8)
                item_mask_eroded = cv.erode(item_mask, kernel, iterations=1)
                total_furniture_mask = cv.bitwise_or(total_furniture_mask, item_mask_eroded)

            # Debug Overlay (Blue)
            blue_layer = np.zeros_like(frame)
            blue_layer[total_furniture_mask == 255] = [255, 0, 0]
            frame = cv.addWeighted(frame, 1.0, blue_layer, 0.3, 0)

            for pet_name, pet_contour in pet_contours:
                pet_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv.fillPoly(pet_mask, [pet_contour], 255)

                intersection = cv.bitwise_and(total_furniture_mask, pet_mask)
                overlap_area = np.count_nonzero(intersection)
                pet_area = np.count_nonzero(pet_mask)

                if pet_area > 0:
                    overlap_ratio = overlap_area / pet_area
                    M = cv.moments(pet_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        txt_color = (0, 0, 255) 
                        if overlap_ratio >= IOU_THRESHOLD:
                            txt_color = (0, 255, 0) 
                        cv.putText(frame, f"{int(overlap_ratio*100)}%", (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

                    if overlap_ratio >= IOU_THRESHOLD:
                        if pet_name == "dog": dog_detected_on_couch = True
                        elif pet_name == "cat": cat_detected_on_couch = True

        if person_present:
            cv.putText(frame, "Human Override", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            dog_detected_on_couch = False
            cat_detected_on_couch = False

        if dog_detected_on_couch:
            current_state = "dog"
            last_valid_animal = "dog"
            consecutive_miss_checks = 0
        elif cat_detected_on_couch:
            current_state = "cat"
            last_valid_animal = "cat"
            consecutive_miss_checks = 0
        else:
            consecutive_miss_checks += 1
            if consecutive_miss_checks < CLEAR_THRESHOLD_CHECKS:
                current_state = last_valid_animal
            else:
                current_state = "clear"
                last_valid_animal = "clear"

        publish_state(current_state, frame)

        cv.imshow("Pet Monitor Seg", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Loop Exception: {e}")
        time.sleep(1)

client.loop_stop()
if 'video_feed' in locals() and video_feed.isOpened():
    video_feed.release()
cv.destroyAllWindows()
