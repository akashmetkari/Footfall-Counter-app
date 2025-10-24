# Footfall-Counter-app
YOLOv8 + DeepSORT based Footfall Counter using Streamlit

# 🧠 Footfall Counter using YOLOv8 + DeepSORT

A real-time people counter app that detects and tracks individuals crossing a virtual line using YOLOv8 (Ultralytics) and DeepSORT tracking — built with Streamlit for an interactive web UI.

## 🚀 Features
- Real-time detection and tracking
- Entry/Exit counting with line crossing
- Downloadable processed video
- Built using free and open-source models

## 🛠️ Tech Stack
- **YOLOv8n** — Object detection
- **DeepSORT** — Multi-object tracking
- **Streamlit** — UI
- **OpenCV** — Video processing


## 🚀 Brief Description of the Approach

1. **Object Detection**:  
   - The YOLOv8 model (`yolov8n.pt`) detects persons in each video frame.
   - Only detections with confidence > 0.5 are considered to reduce false positives.

2. **Object Tracking**:  
   - DeepSORT tracker assigns unique IDs to each detected person and tracks their movement across frames.
   - Tracks are updated in real-time to maintain consistent IDs for accurate counting.

3. **Footfall Counting**:  
   - A horizontal virtual line is drawn in the middle of the frame.
   - The system calculates the vertical movement of each person’s bounding box bottom center (`cy`) between frames.
   - If a person crosses the line from top to bottom, it is counted as an **entry**; if from bottom to top, it is counted as an **exit**.

4. **Annotations**:  
   - Bounding boxes, track IDs, and movement lines are displayed on the video.
   - Entry and exit counts are displayed in real-time.

---

## 📊 Counting Logic 

- **Tracking each person**:  
  Each detected person is assigned a unique ID by DeepSORT. Their position is tracked frame by frame.

- **Direction calculation**:  
  ```python
  direction = current_bottom_y - previous_bottom_y

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

