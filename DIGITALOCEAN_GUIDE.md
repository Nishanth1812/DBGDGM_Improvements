# DigitalOcean GPU Droplet — Training Guide

Everything (checkpoints, logs, pkl cache) is stored on the `trainingresults` volume at `/mnt/trainingresults`.

---

## 0. Push Code to GitHub

First, un-track the data and checkpoint from git (too large for LFS):

```powershell
git rm -r --cached Alzhiemers_Training/data/
git rm --cached Alzhiemers_Training/models/checkpoint_latest.pt
git add .gitignore
git commit -m "Remove data from git, use rsync instead"
git push origin main
```

---

## 1. Create Droplet

- **Image:** Ubuntu 22.04 (CUDA pre-installed)
- **GPU:** RTX 6000 Ada
- **Volume:** attach `trainingresults` before booting

Add your SSH public key:

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

---

## 2. SSH In

```bash
ssh root@<DROPLET_IP>
```

---

## 3. Get the Code + Data

Clone the repo:

```bash
git clone https://github.com/Nishanth1812/DBGDGM_Improvements.git
cd DBGDGM_Improvements
```

Then rsync data and checkpoint from your local machine (Git Bash / WSL):

```bash
rsync -avz --progress \
    H:/Personal/Internships/WeKan/DBGDGM_Improvements/Alzhiemers_Training/data/ \
    root@<DROPLET_IP>:/root/DBGDGM_Improvements/Alzhiemers_Training/data/

rsync -avz \
    H:/Personal/Internships/WeKan/DBGDGM_Improvements/Alzhiemers_Training/models/ \
    root@<DROPLET_IP>:/root/DBGDGM_Improvements/Alzhiemers_Training/models/
```

---

## 4. Set Up Environment (Once)

> Make sure the `trainingresults` volume is attached before running this.

```bash
cd /root/DBGDGM_Improvements
chmod +x Alzhiemers_Training/setup_droplet.sh \
         Alzhiemers_Training/train_do.sh \
         Alzhiemers_Training/sync_results.sh
./Alzhiemers_Training/setup_droplet.sh
source .venv/bin/activate
```

Creates on volume:

- `/mnt/trainingresults/models_oasis_1/` — checkpoints
- `/mnt/trainingresults/logs/` — training logs
- `/mnt/trainingresults/cache/oasis/` — pkl cache (symlinked from `data/oasis/`)

---

## 5. Copy Checkpoint to Volume

```bash
mkdir -p /mnt/trainingresults/models_oasis_1
cp Alzhiemers_Training/models/checkpoint_latest.pt \
   /mnt/trainingresults/models_oasis_1/checkpoint_best_valid.pt
```

---

## 6. Resume Training

```bash
./Alzhiemers_Training/train_do.sh \
    --resume-from /mnt/trainingresults/models_oasis_1/checkpoint_best_valid.pt
```

Automatically runs in a tmux session called `dbgdgm_train`. Check in anytime:

```bash
tmux attach -t dbgdgm_train
# Ctrl+B then D to detach
```

---

## 7. Download Results

```bash
rsync -avz root@<DROPLET_IP>:/mnt/trainingresults/ ./local_results/
```

---

## Notes

- **Destroy the droplet when done** — the volume persists and keeps all results
- To resume on a new droplet: attach the volume, repeat steps 2–4, then step 6
- Expected runtime: ~30–40 min preprocessing + 3–9 hours training
- TF32 + cuDNN benchmark enabled automatically for the RTX 6000 Ada
