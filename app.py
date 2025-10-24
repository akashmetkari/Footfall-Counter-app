import os, cv2, time
import numpy as np
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(page_title="Footfall Counter", layout="wide")
st.title("🧠 Footfall Counter using YOLOv8 + DeepSORT")
st.markdown("Upload a video to count how many people entered or exited across a virtual line.")

video = st.file_uploader("🎥 Upload a video", type=["mp4", "avi", "mov"])

if video:
    with open("input_video.mp4", "wb") as f:
        f.write(video.read())

    st.info("⏳ Processing started... Please wait.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

    # Load model and tracker
    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=40, n_init=3, nn_budget=50, max_iou_distance=0.7)

    cap = cv2.VideoCapture("input_video.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    line_y = height // 2
    entry, exit_ = 0, 0
    last_positions, counted_in, counted_out = {}, set(), set()
    movement_threshold = 10  # reduce flicker counts

    stframe = st.empty()
    progress = st.progress(0)
    start_time = time.time()
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # YOLOv8 detection
        results = model(frame, verbose=False)
        detections = []

        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    if classes[i] == 0 and confs[i] > 0.5:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], confs[i], "person"))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue

            track_id = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            bottom_y = y2

            prev_y = last_positions.get(track_id, bottom_y)
            direction = bottom_y - prev_y

            # Detect line crossing
            if abs(direction) > movement_threshold:
                if prev_y < line_y <= bottom_y and track_id not in counted_in:
                    entry += 1
                    counted_in.add(track_id)
                    counted_out.discard(track_id)
                elif prev_y > line_y >= bottom_y and track_id not in counted_out:
                    exit_ += 1
                    counted_out.add(track_id)
                    counted_in.discard(track_id)

            last_positions[track_id] = bottom_y

            # Draw annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

        # Draw line and counts
        cv2.line(frame, (0, line_y), (width, line_y), (0,0,255), 2)
        cv2.putText(frame, f"IN: {entry}  OUT: {exit_}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR")
        progress.progress(min(frame_no / total_frames, 1.0))

    cap.release()
    out.release()

    end_time = time.time()
    st.success(f"✅ Done! Entries: {entry}, Exits: {exit_} (Processed in {end_time - start_time:.1f}s)")

    with open("output.mp4", "rb") as f:
        st.download_button("📥 Download Processed Video", f, "footfall_output.mp4", "video/mp4")

    st.balloons()
