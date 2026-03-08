# DBGDGM_Improvements

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

### 3. Point the project to the attached volume

The training scripts use `/mnt/trainingresults`, so map that path to the attached `results` volume:

```bash
mkdir -p /mnt/results
ln -sfn /mnt/results /mnt/trainingresults
```

### 4. Run the one-time setup

```bash
chmod +x Alzhiemers_Training/setup_droplet.sh Alzhiemers_Training/train_do.sh
./Alzhiemers_Training/setup_droplet.sh
```

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

The slower baseline, kept here only for comparison, is direct recursive copy:

```powershell
scp -r .\Alzhiemers_Training\data root@159.223.209.77:/root/DBGDGM_Improvements/Alzhiemers_Training/
```

### 6. Start training in tmux

```bash
./Alzhiemers_Training/train_do.sh
```

This starts a fresh training run inside a tmux session named `dbgdgm_train`, so you can disconnect safely.

### 7. Reattach or detach later

```bash
tmux attach -t dbgdgm_train
```

Detach without stopping training with `Ctrl+B`, then `D`.

### 8. Where results are saved

Outputs are written to the attached volume under `/mnt/results`, including checkpoints and logs.