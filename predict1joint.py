# predict1joint.py — single-joint 12s->3s predicting with target-aware anchoring
import argparse, glob, math, random, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ======== PROJECT PATHS ========
BASE = Path(r"C:\Users\saabi\Downloads\single_person_joints")
SEQS_ROOT = BASE / "data" / "sequences"

# Put checkpoints and split-level summaries in dedicated folders:
MODELS_DIR  = BASE / "models"
METRICS_DIR = BASE / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ======== TIMING ========
FPS = 12
T_IN, T_OUT = 12*FPS, 3*FPS
SEQ_LEN = T_IN + T_OUT

# ======== UTILS ========
def set_seed(s=1337):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ema_1d(x, alpha=0.12):
    y = np.empty_like(x, dtype=np.float32)
    if len(x)==0: return x
    y[0]=x[0]
    for i in range(1,len(x)): y[i]=alpha*x[i]+(1-alpha)*y[i-1]
    return y

def _frames_dir_for(seq_dir: Path) -> Path:
    """Return where the raw frames live: seq/original if exists, else seq."""
    return seq_dir/"original" if (seq_dir/"original").exists() else seq_dir

def _read_dims_from_seq_dir(seq_dir: Path):
    frames_dir = _frames_dir_for(seq_dir)
    imgs = sorted(glob.glob(str(frames_dir/"*.jpg")))
    if imgs:
        img = cv2.imread(imgs[0]); h,w = img.shape[:2]; return w,h
    pv = frames_dir/"preview.mp4"
    if pv.exists():
        cap=cv2.VideoCapture(str(pv)); ok,frame=cap.read(); cap.release()
        if ok: h,w=frame.shape[:2]; return w,h
    return 1920,1080

def _joint_xy_series(df_all: pd.DataFrame, j: int):
    sx = df_all[df_all["Joint"]==j].set_index("SeqFrame")["X"].reindex(range(SEQ_LEN))
    sy = df_all[df_all["Joint"]==j].set_index("SeqFrame")["Y"].reindex(range(SEQ_LEN))
    sx = sx.interpolate(limit_direction="both").ffill().bfill()
    sy = sy.interpolate(limit_direction="both").ffill().bfill()
    return sx.to_numpy(np.float32), sy.to_numpy(np.float32)

def _centers_scales(df_all: pd.DataFrame, mode: str, joint_idx: int):
    """
    Returns cx, cy, s arrays (len=SEQ_LEN) per frame.
    mode: 'person' | 'shoulders' | 'head'
    """
    if mode in ("shoulders","head"):
        x5,y5 = _joint_xy_series(df_all, 5)
        x6,y6 = _joint_xy_series(df_all, 6)
        cx = (x5 + x6) * 0.5
        cy = (y5 + y6) * 0.5
        shoulder_w = np.sqrt((x6-x5)**2 + (y6-y5)**2)
        s = np.maximum(shoulder_w * 2.5, 1.0).astype(np.float32)
        if float(np.nanstd(shoulder_w)) >= 1e-3:
            return cx.astype(np.float32), cy.astype(np.float32), s
        # fall back if degenerate
        mode = "person"

    # fallback / 'person': bbox of all joints per frame
    rows=[]
    for t,g in df_all.groupby("SeqFrame"):
        xs=g["X"].to_numpy(np.float32); ys=g["Y"].to_numpy(np.float32)
        if xs.size==0:
            rows.append((int(t), np.nan,np.nan,np.nan)); continue
        xmin,xmax=float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin,ymax=float(np.nanmin(ys)), float(np.nanmax(ys))
        cx=0.5*(xmin+xmax); cy=0.5*(ymin+ymax); s=max(xmax-xmin, ymax-ymin, 1.0)
        rows.append((int(t),cx,cy,s))
    cdf=pd.DataFrame(rows, columns=["SeqFrame","cx","cy","s"]).set_index("SeqFrame")
    cdf=cdf.reindex(range(SEQ_LEN))
    for col in ["cx","cy","s"]:
        cdf[col]=cdf[col].interpolate(limit_direction="both").ffill().bfill()
    return (cdf["cx"].to_numpy(np.float32),
            cdf["cy"].to_numpy(np.float32),
            cdf["s"].to_numpy(np.float32))

def _read_seq_csv_and_frames_dir(seq_dir: Path):
    """Return (df, frames_dir) where frames_dir holds the raw JPG frames."""
    p1 = seq_dir / "sequence.csv"
    if p1.exists():
        return pd.read_csv(p1), _frames_dir_for(seq_dir)
    p2 = seq_dir / "original" / "sequence.csv"
    if p2.exists():
        return pd.read_csv(p2), _frames_dir_for(seq_dir)
    raise FileNotFoundError(f"sequence.csv not found under {seq_dir} (or in original/)")

def _load_one_sequence(csv_path: Path, joint_idx: int, anchor_mode: str):
    df = pd.read_csv(csv_path)
    if "SeqFrame" not in df.columns:
        raise RuntimeError(f"sequence.csv missing SeqFrame: {csv_path}")
    cx,cy,s = _centers_scales(df, anchor_mode, joint_idx)
    xj,yj = _joint_xy_series(df, joint_idx)
    xj, yj = ema_1d(xj), ema_1d(yj)
    seq_dir = csv_path.parent if csv_path.name == "sequence.csv" else csv_path.parent.parent
    w,h = _read_dims_from_seq_dir(seq_dir)
    return xj,yj,cx,cy,s,w,h,seq_dir

# ======== DATASET ========
class JointsSeqDataset(Dataset):
    def __init__(self, root: Path, joint_idx: int, split="train", val_frac=0.2,
                 anchor_mode="person", seq_glob="seq_*", only_seq=None):
        """
        seq_glob: pattern of sequence folders to include (default: all seq_*).
        only_seq: comma-separated names (e.g. "seq_000,seq_003") to restrict.
        """
        self.anchor_mode = anchor_mode

        # gather sequence dirs, then pick CSV from seq/ or seq/original/
        seq_dirs = sorted(root.glob(f"**/{seq_glob}"))
        seq_csvs = []
        for d in seq_dirs:
            p = d / "sequence.csv"
            if not p.exists():
                p = d / "original" / "sequence.csv"
            if p.exists():
                seq_csvs.append(p)

        if only_seq:
            wanted = set([s.strip() for s in only_seq.split(",") if s.strip()])
            seq_csvs = [p for p in seq_csvs if p.parent.name in wanted or p.parent.parent.name in wanted]

        if not seq_csvs:
            raise RuntimeError(f"No sequences found under {root} with pattern '{seq_glob}'"
                               + (f" and filter {only_seq}" if only_seq else ""))

        # deterministic split
        rng = np.random.default_rng(123)
        idxs = np.arange(len(seq_csvs)); rng.shuffle(idxs)
        cut = int(len(idxs)*(1-val_frac))
        keep = idxs[:cut] if split=="train" else idxs[cut:]

        self.items=[]
        for i in keep:
            try:
                x,y,cx,cy,s,w,h,seq_dir = _load_one_sequence(seq_csvs[i], joint_idx, anchor_mode)
            except Exception:
                continue
            if len(x)!=SEQ_LEN: continue
            xn=(x-cx)/s; yn=(y-cy)/s
            self.items.append({
                "x_norm":xn.astype(np.float32), "y_norm":yn.astype(np.float32),
                "cx":cx.astype(np.float32), "cy":cy.astype(np.float32), "s":s.astype(np.float32),
                "w":float(w), "h":float(h), "seq_dir":seq_dir
            })
        if not self.items:
            raise RuntimeError(f"No usable sequences in {split} for joint {joint_idx}")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it=self.items[idx]
        xy = np.stack([it["x_norm"], it["y_norm"]], axis=-1)  # [180,2]
        x_in = xy[:T_IN]
        y_out = xy[T_IN:]
        cx_out = it["cx"][T_IN:]; cy_out = it["cy"][T_IN:]; s_out = it["s"][T_IN:]
        return (torch.from_numpy(x_in), torch.from_numpy(y_out),
                torch.from_numpy(cx_out), torch.from_numpy(cy_out), torch.from_numpy(s_out),
                torch.tensor([it["w"], it["h"]], dtype=torch.float32),
                str(it["seq_dir"]))

# ======== MODEL ========
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers>1 else 0.0,
                               bidirectional=True)
        self.merge_h = nn.Linear(hidden_dim*2, hidden_dim)
        self.merge_c = nn.Linear(hidden_dim*2, hidden_dim)
        self.dec_cell = nn.LSTMCell(2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x_in, y_out=None, teacher_forcing=0.5):
        _, (h, c) = self.encoder(x_in)              # [2*L,B,H]
        h_last = torch.cat([h[-2], h[-1]], dim=1)   # [B,2H]
        c_last = torch.cat([c[-2], c[-1]], dim=1)
        h_t = torch.tanh(self.merge_h(h_last))
        c_t = torch.tanh(self.merge_c(c_last))
        dec_in = x_in[:, -1, :]                     # last observed (x,y)
        outs=[]
        for t in range(T_OUT):
            h_t, c_t = self.dec_cell(dec_in, (h_t, c_t))
            out_t = self.fc(h_t)
            outs.append(out_t.unsqueeze(1))
            if (y_out is not None) and (torch.rand(1).item() < teacher_forcing):
                dec_in = y_out[:, t, :]
            else:
                dec_in = out_t.detach()
        return torch.cat(outs, dim=1)

# ======== TRAIN / EVAL ========
def make_time_weights(tail_boost: float, tail_portion: float, device):
    w = torch.ones(T_OUT, dtype=torch.float32, device=device)
    k = int(T_OUT * max(0.0, min(1.0, tail_portion)))
    if k > 0:
        ramp = torch.linspace(1.0, float(tail_boost), steps=k, device=device)
        w[-k:] = ramp
    return w / w.mean()

def train_one_epoch(model, loader, opt, device, tf_ratio=0.7, clip=1.0, time_w=None):
    model.train()
    crit = nn.SmoothL1Loss(reduction="none", beta=0.02)
    running = 0.0
    for x_in, y_out, *_ in loader:
        x_in=x_in.to(device); y_out=y_out.to(device)
        opt.zero_grad()
        pred = model(x_in, y_out, teacher_forcing=tf_ratio)   # [B,T_OUT,2]
        loss_t = crit(pred, y_out).mean(dim=-1)               # [B,T_OUT]
        loss = (loss_t * time_w.unsqueeze(0)).mean() if time_w is not None else loss_t.mean()
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        running += loss.item() * x_in.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_mse_pixels(model, loader, device, anchor_mode):
    model.eval(); mse_sum=0.0; n=0
    for x_in,y_out,_,_,_,_,seq_dirs in loader:
        x_in=x_in.to(device)
        pred = model(x_in, None, teacher_forcing=0.0).cpu().numpy()  # [B,36,2]
        for i, seq_dir in enumerate(seq_dirs):
            seq_dir = Path(seq_dir)
            df, _frames_dir = _read_seq_csv_and_frames_dir(seq_dir)
            Cx,Cy,S = _centers_scales(df, mode=anchor_mode, joint_idx=0)
            start = SEQ_LEN - T_OUT
            cx = Cx[start:][...,None]; cy = Cy[start:][...,None]; s = S[start:][...,None]
            pred_px = np.concatenate([pred[i, ...,:1]*s + cx, pred[i, ...,1:2]*s + cy], axis=-1)
            gt_norm = y_out[i].cpu().numpy()
            gt_px   = np.concatenate([gt_norm[...,:1]*s + cx, gt_norm[...,1:2]*s + cy], axis=-1)
            mse_sum += float(((pred_px - gt_px)**2).mean())
            n += 1
    return mse_sum/max(1,n)

def _pairwise_l2(a,b): d=a-b; return np.sqrt((d[...,0]**2 + d[...,1]**2))

def _clean_legacy_outputs(seq_dir: Path, joint_idx: int):
    """Move old top-level files into pred_joint<J>/ to avoid confusion."""
    pred_dir = seq_dir / f"pred_joint{joint_idx}"
    pred_dir.mkdir(exist_ok=True)
    legacy = [
        (seq_dir / f"pred_overlay_joint{joint_idx}.mp4", pred_dir / "overlay_legacy.mp4"),
        (seq_dir / f"metrics_joint{joint_idx}.csv",      pred_dir / "metrics_legacy.csv"),
        (seq_dir / f"pred_joint{joint_idx}.csv",         pred_dir / "pred_legacy.csv"),
    ]
    for src, dst in legacy:
        if src.exists() and not dst.exists():
            try: shutil.move(str(src), str(dst))
            except Exception: pass

@torch.no_grad()
def _write_pred_csv(seq_dir: Path, pred_px_seq, joint_idx: int, frame_list):
    start = SEQ_LEN - T_OUT  # 144
    rows=[]
    for k in range(T_OUT):
        seq_frame = start + k
        rows.append({
            "SeqFrame": int(seq_frame),
            "FrameNum": int(frame_list[seq_frame]),
            "PredX": float(pred_px_seq[k,0]),
            "PredY": float(pred_px_seq[k,1]),
        })
    pred_dir = seq_dir / f"pred_joint{joint_idx}"
    pred_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(pred_dir/"pred.csv", index=False)

@torch.no_grad()
def eval_and_visualize(
    model, loader, device, joint_idx,
    viz_limit=12, stash_original=False,
    anchor_mode="person", split_tag="val",
):
    """
    Writes per-sequence:
      seq_dir/pred_joint{J}/overlay.mp4
      seq_dir/pred_joint{J}/metrics.csv
      seq_dir/pred_joint{J}/pred.csv
    And a split-level summary to METRICS_DIR/{split_tag}_metrics_joint{J}.csv
    Optionally moves raw frames+preview+sequence.csv into seq_dir/original/ (stash_original=True)
    """
    model.eval()
    all_rows = []
    made = 0

    for x_in, _, _, _, _, _, seq_dirs in loader:
        x_in = x_in.to(device)
        pred_norm = model(x_in, None, teacher_forcing=0.0).cpu().numpy()  # [B,36,2]

        for i, seq_dir in enumerate(seq_dirs):
            seq_dir = Path(seq_dir)

            # Optionally stash originals once (skip if already stashed)
            if stash_original:
                orig = seq_dir / "original"
                # Only stash if we're clearly at top level (has JPGs and sequence.csv here)
                at_top = (seq_dir / "sequence.csv").exists() and any(seq_dir.glob("*.jpg"))
                if at_top:
                    orig.mkdir(parents=True, exist_ok=True)
                    for p in seq_dir.glob("*.jpg"):
                        shutil.move(str(p), str(orig / p.name))
                    for name in ("preview.mp4", "sequence.csv"):
                        p = seq_dir / name
                        if p.exists():
                            shutil.move(str(p), str(orig / name))
                # else: already stashed → do nothing

            # always read CSV/frames from wherever they live now
            df, frames_dir = _read_seq_csv_and_frames_dir(seq_dir)

            # tidy old outputs (if any)
            _clean_legacy_outputs(seq_dir, joint_idx)

            # compute px anchors + GT
            Cx, Cy, S = _centers_scales(df, mode=anchor_mode, joint_idx=joint_idx)
            xj, yj    = _joint_xy_series(df, joint_idx)
            start     = SEQ_LEN - T_OUT
            gt_px     = np.stack([xj[start:], yj[start:]], axis=-1)  # [36,2]
            cx_h = Cx[start:][..., None]; cy_h = Cy[start:][..., None]; s_h = S[start:][..., None]
            pred_px = np.concatenate(
                [pred_norm[i, ...,:1]*s_h + cx_h, pred_norm[i, ...,1:2]*s_h + cy_h],
                axis=-1
            )

            # metrics
            d = pred_px - gt_px
            dist = np.sqrt(d[...,0]**2 + d[...,1]**2)
            ade, fde = float(dist.mean()), float(dist[-1])
            out_dir = seq_dir / f"pred_joint{joint_idx}"
            out_dir.mkdir(exist_ok=True)
            pd.DataFrame([{"ADE_px": ade, "FDE_px": fde}]).to_csv(out_dir / "metrics.csv", index=False)

            # per-seq predicted coords (keep FrameNum)
            df_frames = df.drop_duplicates(subset=["SeqFrame"])[["SeqFrame","FrameNum"]].sort_values("SeqFrame")
            frame_list = df_frames["FrameNum"].astype(int).tolist()
            _write_pred_csv(seq_dir, pred_px, joint_idx, frame_list)

            # overlay preview
            if made < (viz_limit if viz_limit >= 0 else 10**9):
                img0 = cv2.imread(str(frames_dir / f"{frame_list[0]:06d}.jpg"))
                if img0 is not None:
                    h, w = img0.shape[:2]
                    vw = cv2.VideoWriter(str(out_dir / "overlay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                    col_gt = (255, 255, 0); col_pr = (255, 0, 255); r = 4
                    for t in range(len(frame_list)):
                        img = cv2.imread(str(frames_dir / f"{frame_list[t]:06d}.jpg"))
                        if img is None: continue
                        if t >= start:
                            k = t - start
                            gx, gy = int(gt_px[k, 0]),  int(gt_px[k, 1])
                            px, py = int(pred_px[k, 0]), int(pred_px[k, 1])
                            cv2.circle(img, (gx, gy), r, col_gt, -1, cv2.LINE_AA)
                            cv2.circle(img, (px, py), r, col_pr, -1, cv2.LINE_AA)
                            if t == start:
                                cv2.putText(img, "GT (cyan)  Pred (magenta)", (10, 24),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2)
                        vw.write(img)
                    vw.release()
                    made += 1

            all_rows.append({"seq_dir": str(seq_dir), "ADE_px": ade, "FDE_px": fde})

    out_summary = METRICS_DIR / f"{split_tag}_metrics_joint{joint_idx}.csv"
    pd.DataFrame(all_rows).to_csv(out_summary, index=False)
    if all_rows:
        mean_ade = sum(r["ADE_px"] for r in all_rows)/len(all_rows)
        mean_fde = sum(r["FDE_px"] for r in all_rows)/len(all_rows)
        print(f"\n[{split_tag}] joint {joint_idx}: ADE={mean_ade:.2f}px | FDE={mean_fde:.2f}px")
        print(f"Saved per-seq metrics and summary: {out_summary}")

# -------- Safe resume helper --------
def _safe_autoresume(model: nn.Module, joint_idx: int, device: torch.device, path_override: str = ""):
    ckpt_path = Path(path_override) if path_override else (MODELS_DIR / f"lstm_joint{joint_idx}.pt")
    if not ckpt_path.exists():
        return False
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    # filter to matching shapes
    model_sd = model.state_dict()
    filtered = {k:v for k,v in state.items() if (k in model_sd and tuple(v.shape)==tuple(model_sd[k].shape))}
    missing = [k for k in model_sd.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f"[resume] loaded {len(filtered)} tensors from {ckpt_path.name}; skipped {len(missing)} (shape/name mismatch)")
    return True

# ======== MAIN ========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=str, default="11",
                    help="COCO index '14', comma list '0,5,11', or 'all'")
    ap.add_argument("--viz_all", action="store_true",
                    help="Render previews for every sequence (overrides --viz_limit)")
    ap.add_argument("--render_split", choices=["val","both"], default="val",
                    help="Which split to render predictions for (always trains on 'train').")
    ap.add_argument("--anchor", choices=["person","shoulders","head"], default="person",
                    help="Anchor/scale strategy.")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--tail_boost", type=float, default=2.0)
    ap.add_argument("--tail_portion", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction for split.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--viz_limit", type=int, default=50, help="-1 for all")

    # control which sequences are included
    ap.add_argument("--seq_glob", type=str, default="seq_*",
                    help="Glob relative to data/sequences (e.g. 'angle01/id_001/seq_000', 'angle01/**', 'seq_*')")
    ap.add_argument("--only_seq", type=str, default="",
                    help="Comma list of sequence folder names to keep (e.g. 'seq_000,seq_003')")

    # layout helper
    ap.add_argument("--stash_original", action="store_true",
                    help="Move raw frames/preview/sequence.csv into seq/original/ before writing predictions")

    # resume options
    ap.add_argument("--resume", type=str, default="",
                    help="Path to models/lstm_joint<J>.pt to continue training from")
    ap.add_argument("--auto_resume", action="store_true",
                    help="If set, automatically resume from models/lstm_joint<J>.pt when it exists")

    args = ap.parse_args()

    if args.viz_all:
        args.viz_limit = 10**9  # effectively 'all'

    def _parse_joint_arg(s: str):
        s = s.strip().lower()
        if s == "all":
            return list(range(17))
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [int(p) for p in parts]

    joint_list = _parse_joint_arg(args.joint)

    set_seed(1337)
    device = torch.device(args.device)

    for j in joint_list:
        print(f"\n==== Joint {j} ====\n")

        # Datasets / loaders (respects --seq_glob and --only_seq)
        train_ds = JointsSeqDataset(SEQS_ROOT, j, split="train", val_frac=args.val_frac,
                                    anchor_mode=args.anchor,
                                    seq_glob=args.seq_glob, only_seq=args.only_seq)
        val_ds   = JointsSeqDataset(SEQS_ROOT, j, split="val",   val_frac=args.val_frac,
                                    anchor_mode=args.anchor,
                                    seq_glob=args.seq_glob, only_seq=args.only_seq)
        train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

        # Model / opt / sched
        model = Seq2SeqLSTM(input_dim=2, hidden_dim=args.hidden, num_layers=args.layers).to(device)

        # Resume logic
        resumed = False
        if args.resume:
            print(f"[joint {j}] Resuming from explicit: {args.resume}")
            resumed = _safe_autoresume(model, j, device, path_override=args.resume)
        elif args.auto_resume:
            ck = MODELS_DIR / f"lstm_joint{j}.pt"
            if ck.exists():
                print(f"[joint {j}] Auto-resume from: {ck}")
                resumed = _safe_autoresume(model, j, device, path_override=str(ck))

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
        time_w = make_time_weights(args.tail_boost, args.tail_portion, device)

        # Train
        best = math.inf
        for ep in range(1, args.epochs + 1):
            tf = max(0.0, 0.7 * (1 - (ep - 1) / max(1, args.epochs)))
            tr = train_one_epoch(model, train_ld, opt, device, tf_ratio=tf, clip=1.0, time_w=time_w)
            val = eval_mse_pixels(model, val_ld, device, args.anchor)
            print(f"[joint {j}] Epoch {ep:02d} | train(SmoothL1,norm): {tr:.6f} | val MSE(px): {val:.2f}")
            if val < best:
                best = val
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / f"lstm_joint{j}.pt")
            sched.step()

        # Load best
        try:
            state = torch.load(MODELS_DIR / f"lstm_joint{j}.pt", map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(MODELS_DIR / f"lstm_joint{j}.pt", map_location=device)
        model.load_state_dict(state, strict=False)

        # Render predictions/metrics on chosen split(s)
        if args.render_split in ("val","both"):
            eval_and_visualize(
                model, val_ld, device, j,
                viz_limit=args.viz_limit,
                stash_original=args.stash_original,
                anchor_mode=args.anchor,
                split_tag="val",
            )
        if args.render_split == "both":
            eval_and_visualize(
                model, train_ld, device, j,
                viz_limit=args.viz_limit,
                stash_original=args.stash_original,
                anchor_mode=args.anchor,
                split_tag="train",
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__=="__main__":
    main()
