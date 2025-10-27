import cv2
import threading
import time
from queue import Queue, Empty
from flask import Flask, Response, render_template
from flask_sock import Sock
from demo_detector import detect, check_person_on_object
from message_sender_line import send_msg, send_opencv_frame

app = Flask(__name__)
sock = Sock(app)

frame_lock = threading.Lock()
latest_frame = None
fall_detected_time = None
message_queue = Queue()

# ------------------- Camera Loop -------------------
def camera_loop():
    global latest_frame, fall_detected_time
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_FPS, 15)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (320, 320))
        fall_suspected, info, new_frame = detect(frame)
        if fall_suspected:
            actually_fallen = check_person_on_object(info, new_frame)
        else:
            actually_fallen = False
        with frame_lock:
            latest_frame = new_frame.copy()

        if actually_fallen:
            if fall_detected_time is None:
                fall_detected_time = time.time()
        else:
            fall_detected_time = None

        if fall_detected_time is not None:
            elapsed = time.time() - fall_detected_time
            if elapsed >= 5:
                msg = "FALLDETECTED"
                print("Sent:", msg)
                # Android/Line alert
                send_msg("⚠️ Fall detected — person remained fallen for 5 seconds!")
                send_opencv_frame(frame)
                message_queue.put(msg)
                fall_detected_time = None

        time.sleep(0.02)  # ~50 FPS capture, detection limited by YOLO/Mediapipe

# ------------------- Video feed -------------------
@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------------- WebSocket -------------------
@sock.route('/ws')
def ws(ws):
    try:
        while True:
            try:
                msg = message_queue.get(timeout=1)
                ws.send(msg)
            except Empty:
                continue
    except:
        pass

# ------------------- Main -------------------
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    print("🌐 Flask server starting...")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
