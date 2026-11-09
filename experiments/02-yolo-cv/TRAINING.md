# Training the experiment-02 models on a Runpod GPU pod

You get 9 small models (`models/*.pt`) that run on your Mac afterwards:

| File | Model | Trained on (CI2N folder) | Purpose |
|---|---|---|---|
| `devices.pt` | YOLO11s detector, 31 classes | `device_identification/` | Components, gnd/vdd symbols, ports, hop arcs |
| `jumpers.pt` | YOLO11s-OBB, 3 classes | `jumper_identification/` | Crossings (`cross`), dot junctions (`dot`), T-junctions (`tee`) |
| `orient_*.pt` (7) | YOLO11n-cls | `device_orientation/` | Pin orientation / mirroring for MOS, BJT, amps, diodes, batteries |

Expected cost on one RTX 4090 or A5000-class GPU: roughly **2–3 hours, a few dollars**. These are estimates. The device detector dominates, and early stopping usually ends it before 150 epochs.

---

## 1. Create the pod

1. Runpod → **Pods → Deploy**.
2. GPU: **RTX 4090 (24 GB)** is a good price/speed choice. An A5000, L4 or A40 also works. With a 16 GB GPU, set `batch=8` for `devices` and `jumpers` in `train.py`.
3. Template: **Runpod PyTorch** (any recent 2.x CUDA version). It already includes Python, torch, CUDA and `runpodctl`.
4. Storage: container disk **20 GB**, volume disk **20 GB** mounted at `/workspace` (keeps files if the pod restarts).
5. Deploy, then **Connect → Start Web Terminal** (or SSH, if you added your public key in Runpod settings).

## 2. Upload the code

On your **Mac**, zip the experiment folder without local-only folders:

```bash
cd ~/University/PhD/CircuitResearch/experiments
zip -r 02-yolo-cv.zip 02-yolo-cv -x "02-yolo-cv/.venv/*" "02-yolo-cv/data/*" "02-yolo-cv/runs/*" "02-yolo-cv/results/*" "02-yolo-cv/models/*"
```

Send it with `runpodctl`, which is preinstalled on the pod. Install it on the Mac with `brew install runpod/runpodctl/runpodctl`:

```bash
runpodctl send 02-yolo-cv.zip        # prints a one-time code
```

On the **pod**:

```bash
cd /workspace
runpodctl receive <code-from-the-mac>
unzip -q 02-yolo-cv.zip
```

Other ways to upload: drag-and-drop in the pod's Jupyter file browser, or `scp` if SSH over TCP is enabled.

## 3. Get the dataset on the pod

It is faster to download it directly than to upload it:

```bash
cd /workspace
wget -q -O ci2n.zip https://github.com/LAD021/ci2n_datasets/archive/refs/heads/main.zip
unzip -q ci2n.zip            # -> /workspace/ci2n_datasets-main
```

## 4. Install and prepare

```bash
cd /workspace/02-yolo-cv
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0   # OpenCV runtime libraries
pip install -r requirements-train.txt
python -c "import torch, ultralytics; print(torch.cuda.get_device_name(0), ultralytics.__version__)"

python prepare_data.py --ci2n /workspace/ci2n_datasets-main
```

`prepare_data.py` converts everything to YOLO format under `data/`. It also **removes training images that duplicate a validation circuit**: about 65 images, matching 49 of the 122 validation circuits. They are listed in `data/leakage.json`. Keep this step; without it, experiment-02 scores would be inflated.

## 5. Train

First a 2–3 minute sanity check (1 epoch on 5 % of the data), then delete its outputs:

```bash
python train.py --smoke
rm -rf runs models
```

Full training. Run it under `nohup` (or `tmux`) so it survives a closed browser tab:

```bash
nohup python train.py > train.log 2>&1 &
tail -f train.log            # Ctrl+C stops watching, not training
```

Train jobs individually if you prefer, e.g. `python train.py devices` and then `python train.py jumpers orient_mos ...`. Finished jobs are recorded in `models/manifest.json`, together with their final validation metrics.

Useful checks while it runs:

```bash
nvidia-smi                                   # GPU busy?
ls runs/                                     # one folder per job: curves, confusion matrices
cat models/manifest.json                     # finished jobs and metrics
```

## 6. Download the models

On the **pod**:

```bash
cd /workspace/02-yolo-cv
zip -r models.zip models
zip -r runs_plots.zip runs -i "*.png" "*.csv" "*.yaml"     # optional: training curves for the paper
runpodctl send models.zip
```

On the **Mac**:

```bash
cd ~/University/PhD/CircuitResearch/experiments/02-yolo-cv
runpodctl receive <code>
unzip -o models.zip          # -> models/devices.pt, models/jumpers.pt, models/orient_*.pt
```

Then **stop or terminate the pod** in the Runpod console. A stopped pod still bills for its volume disk.

## 7. Run on the Mac

See [README.md](README.md). In short:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-infer.txt
.venv/bin/python infer.py ../../datasets/ci2n_datasets-main/validation/images/001.png --debug 001_overlay.png
```

The `.pt` files are plain Ultralytics checkpoints. They load on macOS (CPU or the Apple GPU through MPS) with the same `ultralytics` version as the pod (pinned in both requirements files). The torch versions don't need to match.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: libGL.so.1` | `apt-get install -y libgl1 libglib2.0-0` |
| `CUDA out of memory` | Lower `batch` for `devices` / `jumpers` in `train.py` (16 → 8) |
| `No such file .../data/devices.yaml` | Run `prepare_data.py` first, from the `02-yolo-cv` folder |
| Training stopped when the browser closed | Use `nohup ... &` or `tmux` as above. Re-run only the missing jobs: `python train.py jumpers ...` |
| Pod restarted, files gone | Keep everything under `/workspace` (the volume), not in `/root` |
