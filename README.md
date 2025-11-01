# 🧠 YOLOv8 Real-Time Distance Estimation

A real-time computer vision project that *detects humans and estimates their distance* from the camera using *YOLOv8* and *OpenCV*.

---

## 🚀 Features

* 🧍 *Person Detection* — Detects people in live video feed using YOLOv8.
* 📏 *Distance Measurement* — Estimates the distance (in meters) between the camera and each detected person.
* 🎯 *Visual Feedback* — Draws bounding boxes, distance labels, and a connecting line from the camera center to each person.
* ⚡ *Real-Time Processing* — Uses the lightweight YOLOv8n model for fast inference on most devices.

---

## 📸 Demo

> The application draws bounding boxes around people and displays their approximate distance from the camera in real-time.


[Camera View Example]
┌──────────────────────────────────────┐
│   Person 2.4m                        │
│  [Bounding Box + Distance Line]      │
│         <--- 2.4m --->               │
│             (Camera) 🔵              │
└──────────────────────────────────────┘


---

## 🧩 Tech Stack

| Component                | Description                        |
| ------------------------ | ---------------------------------- |
| *YOLOv8 (Ultralytics)* | Real-time object detection model   |
| *OpenCV*               | Image processing and visualization |
| *Python*               | Core programming language          |

---

## ⚙️ Installation

Clone the repository and install dependencies:

bash
git clone https://github.com/<your-username>/YOLOv8-Distance-Estimation.git
cd YOLOv8-Distance-Estimation
pip install ultralytics opencv-python


---

## ▶️ Usage

Run the following command to start the detection:

bash
python distance_estimation.py


* Press *q* to exit the video window.
* You can also replace the webcam input with a video file:

  python
  cap = cv2.VideoCapture("video.mp4")
  

---

## ⚖️ Distance Calculation Logic

The distance is estimated using the *pinhole camera model*:

[
Distance = \frac{Known\ Height \times Focal\ Length}{Bounding\ Box\ Height}
]

* *KNOWN_HEIGHT:* Real-world height of the target object (default = 170 cm for a person)
* *FOCAL_LENGTH:* Calibrated focal length of the camera (default = 600)

To improve accuracy, calibrate FOCAL_LENGTH using your own camera setup.

---

## 🧠 Code Overview

python
model = YOLO("yolov8n.pt")  # Load YOLOv8 nano model
cap = cv2.VideoCapture(0)   # Open webcam

def estimate_distance(bbox_height):
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height

# Draw detection, line, and distance
cv2.rectangle(...)
cv2.line(...)
cv2.putText(...)


---

## 🧰 Customization

| Parameter      | Default | Description                        |
| -------------- | ------- | ---------------------------------- |
| KNOWN_HEIGHT | 170   | Average human height (in cm)       |
| FOCAL_LENGTH | 600   | Camera-specific focal length       |
| conf > 0.4   | -       | Confidence threshold for detection |

---

## 🧑‍💻 Author

**[srijanprasad](https://github.com/<srijanprasad>)**
📧 Feel free to reach out for collaboration or suggestions.

---

## 📜 License

This project is licensed under the *MIT License* — feel free to use and modify it.
