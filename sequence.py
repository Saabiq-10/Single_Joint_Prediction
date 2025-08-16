import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ========= PATHS (set these to your new layout) =========
BASE         = Path(r"C:\Users\saabi\Downloads\single_person_joints")
VIDEO_DIR    = BASE / "data" / "walks"          # original videos (for fallback)
RESULTS_DIR  = BASE / "data" / "pose_results"   # contains overlay.mp4 + pose_depth.csv
OUTPUT_ROOT  = BASE / "data" / "sequences"      # where sequences will be saved
# ========================================================

FPS = 12
SEQ_LEN = 15 * FPS  # 15s * 12Hz = 180 frames

def load_pose_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize column names
    df["FrameNum"] = df["frame"].astype(str).str.extract(r"(\d+)").astype(int)
    df = df.rename(columns={"id":"TrackID", "joint":"Joint", "x":"X", "y":"Y", "depth":"Depth"})
    return df

def contiguous_runs(nums):
    """Return list of lists of contiguous integers, longest first."""
    s = sorted(set(nums))
    if not s: return []
    runs, cur = [], [s[0]]
    for a,b in zip(s, s[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    runs.sort(key=len, reverse=True)
    return runs

def non_overlapping_blocks(frame_nums, seq_len):
    """
    Yield all non-overlapping contiguous windows of length seq_len.
    If a run is longer than seq_len, it will be split into chunks of size seq_len.
    """
    for run in contiguous_runs(frame_nums):
        if len(run) < seq_len:
            continue
        # step by seq_len to avoid overlap
        for start in range(0, len(run) - seq_len + 1, seq_len):
            yield run[start:start + seq_len]

def open_matching_video(stem: str, prefer_overlay: bool = True) -> Path | None:
    """Find overlay.mp4 (preferred) or original video with same stem."""
    overlay = RESULTS_DIR / stem / "overlay.mp4"
    if prefer_overlay and overlay.exists():
        return overlay
    # fallback to original camera file
    for p in VIDEO_DIR.iterdir():
        if p.stem == stem and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            return p
    return None

def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None

def process_one(stem_dir: Path):
    csv_path = stem_dir / "pose_depth.csv"
    if not csv_path.exists():
        return
    stem = stem_dir.name

    # choose video (overlay preferred so you see the skeleton from pose.py)
    video_path = open_matching_video(stem, prefer_overlay=True)
    if video_path is None:
        print(f"⚠️  No video found for {stem} (overlay or original). Skipping.")
        return

    df = load_pose_csv(csv_path)
    if df.empty:
        return

    out_base = OUTPUT_ROOT / stem
    out_base.mkdir(parents=True, exist_ok=True)

    # open video to copy frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  Could not open {video_path}")
        return

    for track_id, g in tqdm(df.groupby("TrackID"), desc=f"{stem}: tracks"):
        # all frames where this track has at least one joint recorded
        frames = g["FrameNum"].unique().tolist()
        blocks = list(non_overlapping_blocks(frames, SEQ_LEN))
        if not blocks:
            continue

        for seq_idx, block in enumerate(blocks):
            seq_dir = out_base / f"id_{int(track_id):03d}" / f"seq_{seq_idx:03d}"
            seq_dir.mkdir(parents=True, exist_ok=True)

            # write a preview video with the exact 15s chunk (no extra overlay)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            first_img = read_frame(cap, block[0])
            if first_img is None:
                continue
            h, w = first_img.shape[:2]
            vw = cv2.VideoWriter(str(seq_dir / "preview.mp4"), fourcc, FPS, (w, h))

            # filter joints to the chosen window
            g_block = g[g["FrameNum"].isin(block)].copy()
            # map absolute frame to sequence index 0..(SEQ_LEN-1)
            frame_to_idx = {f:i for i, f in enumerate(block)}
            g_block["SeqFrame"] = g_block["FrameNum"].map(frame_to_idx)
            g_block["TimeSec"]  = g_block["FrameNum"] / float(FPS)

            # save frames (pulled straight from overlay/original video)
            for f in block:
                img = read_frame(cap, f)
                if img is not None:
                    cv2.imwrite(str(seq_dir / f"{f:06d}.jpg"), img)
                    vw.write(img)

            vw.release()
            # save the per-joint CSV for this sequence
            cols = ["TrackID","SeqFrame","FrameNum","TimeSec","Joint","X","Y","Depth"]
            g_block = g_block[cols].sort_values(["SeqFrame","Joint"])
            g_block.to_csv(seq_dir / "sequence.csv", index=False)

    cap.release()

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    if not result_dirs:
        print(f"No result folders in {RESULTS_DIR}")
        return
    for d in result_dirs:
        process_one(d)
    print(f"\n✅ Done. Sequences saved in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
