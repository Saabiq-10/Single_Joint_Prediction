Tracking → Sequences → One Joint Training

Simple Setup (needs pytorch, see details below)

# 1) Create env and install
python -m venv venv
venv\Scripts\activate           # or source venv/bin/activate (mac/linux)
pip install -r requirements.txt
# Install PyTorch (choose ONE):
# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# or CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Put your test videos here:
data/videos/

# 3) Run tracking + pose
python pose.py

# 4) Make 15s sequences at 12 Hz
python sequences.py

# 5) Predict one joint (example: left hip=11)
python predict1joint.py --joint 11 --anchor person --viz_all

Recommended: Continue from trained LSTM checkpoints
python predict1joint.py --joint all --hidden 192 --layers 2 --epochs 10 --anchor person --auto_resume --viz_all

Quick tips
Anchors: head/arms → --anchor shoulders; hips/legs → --anchor person.


COCO joints: 0 nose, 1 L_eye, 2 R_eye, 3 L_ear, 4 R_ear, 5 L_sh, 6 R_sh, 7 L_elb, 8 R_elb, 9 L_wr, 10 R_wr, 11 L_hip, 12 R_hip, 13 L_knee, 14 R_knee, 15 L_ank, 16 R_ank.


Outputs for prediction per sequence:

 data/sequences/.../seq_XXX/pred_jointJ/
  pred.csv      # predicted (x,y) for frames 144–179
  metrics.csv   # ADE/FDE
  overlay.mp4   # GT (cyan) vs Pred (magenta)
