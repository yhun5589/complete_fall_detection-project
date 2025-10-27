import cv2
from ultralytics import YOLO
import mediapipe as mp

# --- Load YOLO model ---
model = YOLO('yolo12n.pt')  # GPU auto if available
model.fuse()  # fuse layers for faster CPU inference

# --- Load class names ---
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# --- Mediapipe Pose setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)

TOTAL_LANDMARKS = 33
KEYPOINT_VISIBILITY_THRESHOLD = 0.35  # per-landmark confidence
BODY_VISIBILITY_THRESHOLD = 0.5       # require ≥50% of keypoints inside bbox

# ---------------- Detect Function ----------------
def detect(frame, conf_threshold=0.6):
    info = {}
    new_frame = frame.copy()
    fall_suspected = False

    # --- Run YOLO ---
    results = model(frame, verbose=False, imgsz=320)  # smaller size for speed

    # --- Run Mediapipe ONCE per frame ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    keypoints = []
    if pose_results.pose_landmarks:
        keypoints = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]), lm.visibility)
                     for lm in pose_results.pose_landmarks.landmark]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_detect = classnames[class_id]

            if conf < conf_threshold or class_detect not in ['person', 'sofa', 'bed', 'chair']:
                continue

            width, height = x2 - x1, y2 - y1

            # --- Person body visibility check ---
            if class_detect == 'person':
                # Count keypoints inside bbox
                visible_kps = [kp for kp in keypoints
                               if kp[2] >= KEYPOINT_VISIBILITY_THRESHOLD and x1 <= kp[0] <= x2 and y1 <= kp[1] <= y2]
                kp_fraction = len(visible_kps) / TOTAL_LANDMARKS
                if kp_fraction < BODY_VISIBILITY_THRESHOLD:  # only >50% visible
                    continue

            # --- Store detection info ---
            if class_detect not in info:
                info[class_detect] = []
            info[class_detect].append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": width, "height": height,
            })

            # --- Draw boxes ---
            color = (0, 255, 0) if class_detect != "person" else (255, 255, 0)
            if class_detect != "person":
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(new_frame, f"{class_detect} {int(conf*100)}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Lying heuristic ---
            if class_detect == 'person':
                ratio = height / (width + 1e-6)
                if ratio < 1.15:
                    fall_suspected = True
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(new_frame, "fall?", (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    fall_suspected = False
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(new_frame, "standing", (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return fall_suspected, info, new_frame

# ---------------- Check person on object ----------------
def check_person_on_object(info, frame, iou_threshold=0.1, vertical_tolerance=0.35):
    actually_fall = True
    def box_iou(a, b):
        xA = max(a["x1"], b["x1"])
        yA = max(a["y1"], b["y1"])
        xB = min(a["x2"], b["x2"])
        yB = min(a["y2"], b["y2"])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        if inter_area == 0:
            return 0
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        return inter_area / float(area_a + area_b - inter_area)

    if 'person' not in info:
        return False

    actually_fall = True

    for person in info['person']:
        px1, py1, px2, py2 = person['x1'], person['y1'], person['x2'], person['y2']
        person_bottom_y = py2

        for obj_name in ['bed', 'sofa', 'chair']:
            if obj_name not in info:
                continue

            for obj in info[obj_name]:
                ox1, oy1, ox2, oy2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']

                # Draw object
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 255, 0), 2)
                cv2.putText(frame, obj_name, (ox1, oy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                overlap = box_iou(person, obj)
                horizontal_overlap = max(0, min(px2, ox2) - max(px1, ox1))
                horizontal_fraction = horizontal_overlap / (px2 - px1)
                vertical_position_ok = person_bottom_y <= oy2 and person_bottom_y >= oy1 - obj['height'] * vertical_tolerance

                if overlap > iou_threshold and horizontal_fraction > 0.5 and vertical_position_ok:
                    actually_fall = False
                    cv2.putText(frame, f"On {obj_name}", (px1, py1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break
            if not actually_fall:
                break

        if actually_fall:
            cv2.putText(frame, "FALL DETECTED", (px1, py1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return actually_fall

# ------------------- Test -------------------
if __name__ == "__main__":
    image = cv2.imread('Results-of-Falling.jpg')
    image = cv2.resize(image, (640, 480))
    fall_suspected, info, new_frame = detect(image)
    result = check_person_on_object(info, new_frame)
    print("Actually fallen:", result)
    cv2.imshow("Detection", new_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
