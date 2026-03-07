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

### 5. Copy data and starting checkpoints to the droplet

From your local machine, sync the dataset and any checkpoint you want to resume from:

```bash
rsync -avz --progress \
	H:/Personal/Internships/WeKan/DBGDGM_Improvements/Alzhiemers_Training/data/ \
	root@159.223.209.77:/root/DBGDGM_Improvements/Alzhiemers_Training/data/

rsync -avz --progress \
	H:/Personal/Internships/WeKan/DBGDGM_Improvements/Alzhiemers_Training/models/ \
	root@159.223.209.77:/root/DBGDGM_Improvements/Alzhiemers_Training/models/
```

If resuming from an existing checkpoint, place it on the volume first:

```bash
mkdir -p /mnt/results/models_oasis_1
cp Alzhiemers_Training/models/checkpoint_latest.pt /mnt/results/models_oasis_1/checkpoint_best_valid.pt
```

### 6. Start training in tmux

```bash
./Alzhiemers_Training/train_do.sh \
	--resume-from /mnt/results/models_oasis_1/checkpoint_best_valid.pt
```

This launches training inside a tmux session named `dbgdgm_train`, so you can disconnect safely.

### 7. Reattach or detach later

```bash
tmux attach -t dbgdgm_train
```

Detach without stopping training with `Ctrl+B`, then `D`.

### 8. Where results are saved

Outputs are written to the attached volume under `/mnt/results`, including checkpoints and logs.