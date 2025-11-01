from ultralytics import YOLO
import cv2
import math

# Load YOLOv8n (nano model for speed)
model = YOLO("yolov8n.pt")

# Open webcam (or replace 0 with a video path like 'video.mp4')
cap = cv2.VideoCapture(0)

# Define focal length and reference height (for distance estimation)
# Adjust these values for more accuracy based on your camera setup
KNOWN_HEIGHT = 170  # average person height in cm
FOCAL_LENGTH = 600  # experimental value (you can calibrate this)

def estimate_distance(bbox_height):
    """Estimate distance based on bounding box height"""
    if bbox_height == 0:
        return 0
    distance_cm = (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height
    return round(distance_cm / 100, 2)  # return meters

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    # Get frame center
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Only detect people (COCO class 0 = person)
            if cls == 0 and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_height = y2 - y1
                distance = estimate_distance(bbox_height)

                # Person center point
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {distance}m"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw line from camera center to person
                cv2.line(frame, frame_center, person_center, (0, 0, 255), 2)
                # Display the distance near the midpoint of the line
                mid_x = (frame_center[0] + person_center[0]) // 2
                mid_y = (frame_center[1] + person_center[1]) // 2
                cv2.putText(frame, f"{distance}m", (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw camera center point
    cv2.circle(frame, frame_center, 5, (255, 0, 0), -1)
    cv2.putText(frame, "Camera", (frame_center[0] - 30, frame_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8n - Distance Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




