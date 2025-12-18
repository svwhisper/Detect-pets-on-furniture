import cv2
import json


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

rtsp_url = config.get("rtsp_url", "")

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()