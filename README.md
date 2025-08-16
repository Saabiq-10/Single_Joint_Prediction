# Pose Tracking → Split into Training Sequences → One Joint Training

Simple Setup (needs pytorch, see details below)

# 1) Create venv and install
```
python -m venv venv
venv\Scripts\activate           # or source venv/bin/activate (mac/linux)
pip install -r requirements.txt
```
Install PyTorch (choose ONE):
```
GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
or CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
# 2) Put your test videos here:
```data/videos/```

# 3) Run tracking + pose
```python pose.py```

# 4) Make 15s sequences at 12 Hz
```python sequences.py```

# 5) Train & predict (optionally continue from a saved checkpoint)

**What’s a checkpoint?** After training, a file like models/lstm_joint<J>.pt is saved (one per joint).
Run with ```--auto_resume``` to keep training from that file next time.

Model sizes:

**128×2** = hidden size 128, 2 LSTM layers (default)

**192×2** = larger model (higher capacity)

**Heads up:** the latest checkpoints in `models/` were trained with **192x2**   
To continue training from them, include `--hidden 192 -- layers 2` in your command.  
If you run with default **128x2**, the shapes won't match and the script will start a fresh 128x2 model (creating a seperate set of checkpoints).

**Single joint (example: left hip = 11)**
Resumes if a matching checkpoint exists; remove ```--auto_resume``` to start fresh.

```python predict1joint.py --joint 11 --anchor person --auto_resume --viz_all```

**All joints — standard size (128×2)**
~2× faster and ~1.7–2.2× lower GPU/CPU memory. Starts fresh unless 128x2 checkpoints already exist. 

```python predict1joint.py --joint all --anchor person --epochs 10 --auto_resume --viz_all```

**All joints — high capacity (192×2)**
First run creates 192×2 checkpoints:

```python predict1joint.py --joint all --anchor person --hidden 192 --layers 2 --epochs 35 --viz_all```

Then you can "top up" training:

```python predict1joint.py --joint all --anchor person --hidden 192 --layers 2 --epochs 10 --auto_resume --viz_all```

# Notes

- ```--auto_resume``` looks for ```models/lstm_joint<J>.pt``` and resumes **only** if the shapes match.

- Switching sizes (128×2 vs 192×2)  makes a seperate set of checkpoints.
- Bigger models can help head/wrist/ankle joints but take longer.

# Quick tips
Anchors: head/arms → --anchor shoulders;   
hips/legs → --anchor person.


COCO joints: 0 nose, 1 L_eye, 2 R_eye, 3 L_ear, 4 R_ear, 5 L_sh, 6 R_sh, 7 L_elb, 8 R_elb, 9 L_wr, 10 R_wr, 11 L_hip, 12 R_hip, 13 L_knee, 14 R_knee, 15 L_ank, 16 R_ank.


# Outputs for prediction per sequence:


     
     data/sequences/.../seq_XXX/pred_jointJ/ <br>
       pred.csv      # predicted (x,y) for frames 144–179
       metrics.csv   # ADE/FDE
       overlay.mp4   # GT (cyan) vs Pred (magenta)
