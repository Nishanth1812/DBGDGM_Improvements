# DBGDGM_Improvements

## Kaggle Training

For the end-to-end Kaggle path from raw preprocessing outputs to final `fmri.npy`, `smri.npy`, and `labels.npy`, use `KAGGLE_README.md`.

## DigitalOcean Training Quick Guide

Use this when training on the DigitalOcean droplet at `159.223.209.77` and storing outputs on the attached volume named `results`.

### 1. SSH into the droplet

```bash
ssh root@159.223.209.77
```

### 2. Clone the repo and enter it

```bash
git clone https://github.com/Nishanth1812/DBGDGM_Improvements.git
cd DBGDGM_Improvements
```

If the repo is already present on the droplet, update it instead:

```bash
cd /root/DBGDGM_Improvements
git pull origin main
```

If you have just pulled the latest changes, continue with the next steps below in order. The code update alone is not enough because the droplet still needs the raw OASIS dataset and a fresh cache.

### 3. Point the project to the attached volume

The training scripts use `/mnt/trainingresults`, so map that path to the attached `results` volume:

```bash
mkdir -p /mnt/results
ln -sfn /mnt/results /mnt/trainingresults
```

### 4. Run the setup

```bash
chmod +x Alzhiemers_Training/setup_droplet.sh Alzhiemers_Training/train_do.sh
./Alzhiemers_Training/setup_droplet.sh
```

Run this after pulling new code as well, not just on the first droplet boot.

### 5. Copy the dataset from Windows PowerShell with fast SCP

Use PowerShell on Windows, not WSL. The fast path is to create one tar archive locally and upload that single archive with `scp`. That avoids the file-by-file overhead of `scp -r`, which is the main reason recursive SCP feels slow on large datasets.

Requirements on Windows:

- `scp.exe` from the OpenSSH Client feature
- `tar.exe` available in PowerShell

Fast upload and remote extract in one step:

```powershell
.\Alzhiemers_Training\upload_data_scp.ps1 `
  159.223.209.77 `
  -Identity "$env:USERPROFILE\.ssh\id_ed25519" `
  -ExtractRemote
```

If your SSH key is already the default one used by OpenSSH, run:

```powershell
.\Alzhiemers_Training\upload_data_scp.ps1 159.223.209.77 -ExtractRemote
```

If you want the copy step itself to stay strictly `scp` only, run without `-ExtractRemote`:

```powershell
.\Alzhiemers_Training\upload_data_scp.ps1 `
  159.223.209.77 `
  -Identity "$env:USERPROFILE\.ssh\id_ed25519"
```

That command uploads one tarball with `scp` and then prints the exact `ssh` command needed to extract it on the droplet.

If you have already pulled the latest code to the droplet, this dataset upload is the next required step.

The slower baseline, kept here only for comparison, is direct recursive copy:

```powershell
scp -r .\Alzhiemers_Training\data root@159.223.209.77:/root/DBGDGM_Improvements/Alzhiemers_Training/
```

### 6. Reset old OASIS artifacts before a fresh run

The corrected pipeline now builds subject-level OASIS samples. Old scan-level caches and checkpoints are not compatible with the new code, so do not resume them.

On the droplet, remove the old OASIS cache and move old checkpoints out of the way before training:

```bash
mkdir -p /mnt/trainingresults/backup_pre_subject_fix
mv /mnt/trainingresults/models_oasis_1 /mnt/trainingresults/backup_pre_subject_fix/ 2>/dev/null || true
rm -f /mnt/trainingresults/cache/oasis/oasis*.pkl 2>/dev/null || true
```

After a fresh `git pull`, always do this cleanup before training if the droplet has older OASIS outputs.

Verify the raw dataset exists in the repo path after upload:

```bash
find /root/DBGDGM_Improvements/Alzhiemers_Training/data -maxdepth 1 -type d
```

You should see the four class folders:

- `Non Demented`
- `Very mild Dementia`
- `Mild Dementia`
- `Moderate Dementia`

If those folders are missing, do not start training yet.

### 7. Start training in tmux

```bash
./Alzhiemers_Training/train_do.sh
```

This starts a fresh training run inside a tmux session named `dbgdgm_train`, so you can disconnect safely.

Do not pass `--resume-from` with checkpoints from the previous pipeline.

This is the correct command to run immediately after:

1. `git pull origin main`
2. dataset upload
3. cache/checkpoint cleanup

### 8. Run inference for realistic diagnosis metrics

After training completes, run inference to save embeddings and compute subject-level downstream diagnosis metrics:

```bash
cd /root/DBGDGM_Improvements/Alzhiemers_Training
/root/DBGDGM_Improvements/.venv/bin/python main.py \
  --dataset oasis \
  --categorical-dim 3 \
  --trial 1 \
  --gpu 0 \
  inference
```

This writes `results_inference.npy` and prints diagnosis metrics computed from subject embeddings with repeated stratified evaluation.

Run this only after training has finished successfully.

### 9. Reattach or detach later

```bash
tmux attach -t dbgdgm_train
```

Detach without stopping training with `Ctrl+B`, then `D`.

### 10. Where results are saved

Outputs are written to the attached volume under `/mnt/results`, including checkpoints, logs, and the OASIS cache.

### 11. Download results to a named local run folder

On Windows PowerShell, use the sync helper to pull the full `/mnt/trainingresults` volume directly into a target folder under `local_results`:

```powershell
.\Alzhiemers_Training\sync_results.ps1 `
  159.223.209.77 `
  -Destination ".\local_results\run 2"
```

That extracts the remote results directly into `local_results/run 2` rather than creating an extra nested `trainingresults` folder.