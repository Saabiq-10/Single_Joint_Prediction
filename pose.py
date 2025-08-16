import os
import sys
import time
from pathlib import Path


# If you cloned Depth-Anything, add it here so 'import depth_anything' works.
# If you didn't clone it, leave as-is; the script will run with depth off.
sys.path.insert(0, r"C:\Users\saabi\Downloads\single_person_joints\Depth-Anything")

import cv2
import csv
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque

# drawing + smoothing params
JOINT_RADIUS     = 5          # bigger dots
LIMB_THICKNESS   = 3          # thicker lines
SMOOTH_N         = 1          # rolling window length (frames) per joint
KPT_CONF_THRESH  = 0.0       # ignore very low-confidence keypoints

# COCO-17 skeleton (Ultralytics order)
SKELETON = [
    (5, 6),       # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (11, 12),          # hips
    (5, 11), (6, 12),  # torso diagonals
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16),# right leg
    (0, 1), (0, 2), (1, 3), (2, 4)  # face / head
]

# per-track, per-joint rolling buffers for smoothing
kp_hist = defaultdict(lambda: [deque(maxlen=SMOOTH_N) for _ in range(17)])


# Depth (optional) 
USE_DEPTH = False  # set True if you actually want depth now (slower)
try:
    if USE_DEPTH:
        from depth_anything.dpt import DepthAnything
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import torchvision.transforms as T
except Exception:
    USE_DEPTH = False

# ========= CONFIG (edit these two) =========
VIDEO_INPUT_DIR  = r"C:\Users\saabi\Downloads\single_person_joints\data\videos"           # your videos folder
OUTPUT_BASE_PATH = r"C:\Users\saabi\Downloads\single_person_joints\pose_results"    # outputs per video

# Speed/quality knobs
DET_CONF_THRESHOLD = 0.5
DET_EVERY = 4              # run detector every N frames, reuse between (3–5 is typical)
IMG_SIZE_DET = 448         # detector inference size (reduce if still slow)
IMG_SIZE_POSE = 256        # pose inference size (reduce if still slow)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = (DEVICE == "cuda")  # half-precision only on GPU

os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# YOLO MODELS
print("Loading YOLO models (auto-download if missing)...")
detector   = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")

# Optional warmup (helps stabilize first-run latency on CUDA)
if DEVICE == "cuda":
    detector.predict(np.zeros((640,640,3), dtype=np.uint8), imgsz=IMG_SIZE_DET, verbose=False, half=HALF, device=DEVICE)
    pose_model.predict([np.zeros((256,256,3), dtype=np.uint8)], imgsz=IMG_SIZE_POSE, verbose=False, half=HALF, device=DEVICE)

# ==== Depth-Anything (optional) ====
if USE_DEPTH:
    try:
        DEPTH_BACKBONE = "vitl"  # or 'vitb'/'vits'
        REPO_ID = {
            "vitl": "LiheYoung/depth-anything-large-hf",
            "vitb": "LiheYoung/depth-anything-base-hf",
            "vits": "LiheYoung/depth-anything-small-hf",
        }[DEPTH_BACKBONE]

        ckpt_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")

        depth_model = None
        last_err = None
        for kwargs in (
            {"backbone": DEPTH_BACKBONE, "checkpoint_path": ckpt_path},
            {"encoder":  DEPTH_BACKBONE, "checkpoint_path": ckpt_path},
        ):
            try:
                depth_model = DepthAnything.from_pretrained(**kwargs)
                break
            except Exception as e:
                last_err = e

        if depth_model is None:
            try:
                depth_model = DepthAnything(REPO_ID)  # some forks accept single-arg
            except Exception as e:
                last_err = e
                raise RuntimeError(f"Depth init failed: {last_err}")

        depth_model = depth_model.to(DEVICE).eval()
        depth_transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])
        print("[Depth] Ready.")
    except Exception as e:
        print(f"[Depth] Disabled: {e}")
        USE_DEPTH = False
else:
    depth_model = None
    depth_transform = None

#tracker 
tracker = DeepSort(max_age=10, nn_budget=50)

# Color helper 
_id2color = {}
rng = np.random.default_rng(12345)
def color_for_id(tid):
    if tid not in _id2color:
        _id2color[tid] = tuple(int(x) for x in rng.integers(0, 255, size=3))
    return _id2color[tid]

def run_video(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {video_path}")

    csv_path = out_dir / "pose_depth.csv"
    with open(csv_path, "w", newline="") as csv_fp:
        writer = csv.writer(csv_fp)
        # one row per joint
        writer.writerow(["frame", "id", "joint", "x", "y", "depth"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = None
        frame_idx = 0

        prev_dets = []  # cached detections between detector calls
        t0 = time.time()
        frames_done = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # --- Detect people (sparse) ---
            if frame_idx % DET_EVERY == 0:
                det = detector.predict(frame, imgsz=IMG_SIZE_DET, verbose=False, half=HALF, device=DEVICE)[0]
                if det.boxes is not None:
                    boxes   = det.boxes.xyxy.cpu().numpy()
                    confs   = det.boxes.conf.cpu().numpy()
                    classes = det.boxes.cls.cpu().numpy()
                else:
                    boxes = np.empty((0,4)); confs = np.empty((0,)); classes = np.empty((0,))

                peds = [(b, c) for b, cls, c in zip(boxes, classes, confs) if cls == 0 and c >= DET_CONF_THRESHOLD]
                detections = [([float(x1), float(y1), float(x2-x1), float(y2-y1)], float(c), 0)
                              for (x1, y1, x2, y2), c in peds]
                prev_dets = detections
            else:
                detections = prev_dets

            tracks = tracker.update_tracks(detections, frame=frame)

            # Depth (optional, full frame) 
            if USE_DEPTH:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tens = depth_transform(rgb).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    depth = depth_model(tens)[0, 0].detach().cpu().numpy()
                depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            else:
                depth_resized = None

            # Pose per confirmed track
            for t in tracks:
                if not t.is_confirmed():
                    continue

                x1, y1, x2, y2 = map(int, t.to_ltrb())
                x1c = max(0, x1); y1c = max(0, y1)
                x2c = min(frame.shape[1], x2); y2c = min(frame.shape[0], y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                roi = frame[y1c:y2c, x1c:x2c]
                if roi.size == 0:
                    continue

                # run pose on ROI (pick a modest imgsz for stability vs speed)
                pres = pose_model.predict(
                    roi,
                    imgsz=IMG_SIZE_POSE,   # uses 256 setting
                    conf=0.01,             # be permissive
                    verbose=False,
                    device=DEVICE,
                    half=(DEVICE=="cuda")
                )[0]

                if pres.keypoints is None or pres.keypoints.data is None:
                    continue

                kps_all = pres.keypoints.data.cpu().numpy()  # [N,17,3]
                if kps_all.shape[0] == 0:
                    continue

                # if multiple persons returned in the ROI, pick the one with highest mean conf
                if kps_all.shape[0] > 1:
                    means = np.nanmean(kps_all[:, :, 2], axis=1)
                    kp = kps_all[np.nanargmax(means)]
                else:
                    kp = kps_all[0]  # [17,3]

                # absolute coords + confidence gate
                abs_xy = []
                for j_idx, (xr, yr, c) in enumerate(kp):
                    if c < KPT_CONF_THRESH:
                        abs_xy.append(None)         # mark as missing
                        continue
                    xf = int(x1c + xr)
                    yf = int(y1c + yr)
                    if 0 <= xf < frame.shape[1] and 0 <= yf < frame.shape[0]:
                        abs_xy.append((xf, yf))
                    else:
                        abs_xy.append(None)

                # smoothing per-joint using rolling mean over last SMOOTH_N frames
                hist = kp_hist[t.track_id]
                smoothed_xy = []
                for j_idx, xy in enumerate(abs_xy):
                    if xy is not None:
                        hist[j_idx].append(xy)
                    # if xy is None but we have history, keep previous mean (hold)
                    if len(hist[j_idx]) > 0:
                        xs, ys = zip(*hist[j_idx])
                        smoothed_xy.append((int(np.mean(xs)), int(np.mean(ys))))
                    else:
                        smoothed_xy.append(None)

                col = color_for_id(t.track_id)
                drawn_kps = 0
                # draw limbs first (lines)
                for a, b in SKELETON:
                    pa = smoothed_xy[a]
                    pb = smoothed_xy[b]
                    if pa is None or pb is None:
                        continue
                    cv2.line(frame, pa, pb, col, LIMB_THICKNESS, lineType=cv2.LINE_AA)

                # draw joints (circles) and write CSV
                for j_idx, p in enumerate(smoothed_xy):
                    if p is None:
                        continue
                    cv2.circle(frame, p, JOINT_RADIUS, col, -1, lineType=cv2.LINE_AA)
                    drawn_kps += 1
                    dval = float(depth_resized[p[1], p[0]]) if USE_DEPTH else -1.0
                    writer.writerow([f"{frame_idx:06d}.jpg", t.track_id, j_idx, p[0], p[1], dval])
                    

                # track box + ID
                cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), col, 2)
                cv2.putText(frame, f"ID:{t.track_id}", (x1c, y1c - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


            if out_vid is None:
                h, w = frame.shape[:2]
                out_vid = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps, (w, h))
            out_vid.write(frame)

            frame_idx += 1
            frames_done += 1
            if frames_done % 60 == 0:
                dt = time.time() - t0
                print(f"~{frames_done/dt:.1f} FPS (running)")

        cap.release()
        if out_vid is not None:
            out_vid.release()
    if frames_done % 60 == 0:
        print(f"~{frames_done/dt:.1f} FPS (running) | drawn_kps last frame: {drawn_kps}")
    print(f"Saved video: {out_dir/'overlay.mp4'}")
    print(f"Saved CSV:   {csv_path}")

def main():
    in_dir = Path(VIDEO_INPUT_DIR)
    vids = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    if not vids:
        print(f"No videos found in {VIDEO_INPUT_DIR}")
        return
    for vp in vids:
        out_dir = Path(OUTPUT_BASE_PATH) / vp.stem
        run_video(vp, out_dir)

if __name__ == "__main__":
    main()
