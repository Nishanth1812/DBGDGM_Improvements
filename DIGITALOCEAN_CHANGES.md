# DigitalOcean Documentation Changes

## What Changed

- Added a concise DigitalOcean training guide to the README.
- Updated the droplet target to `159.223.209.77`.
- Documented using the attached `results` volume via `/mnt/results`.
- Added the compatibility step `ln -sfn /mnt/results /mnt/trainingresults` because the existing training scripts still write to `/mnt/trainingresults`.
- Documented the existing tmux workflow with `./Alzhiemers_Training/train_do.sh` and `tmux attach -t dbgdgm_train`.
- Clarified where checkpoints and logs are saved on the attached volume.
- Replaced the transfer example with `scp` commands that work from WSL.
- Changed the training example to start from scratch instead of resuming from an existing checkpoint.

## Why This Is Better

- It matches your current droplet and storage setup instead of the older IP and older wording.
- It avoids changing the training scripts just to support a different mount point; the symlink keeps the current scripts working as-is.
- It makes result persistence explicit, so checkpoints and logs survive droplet restarts as long as the `results` volume stays attached.
- It uses tmux through the existing launcher script, which means training continues after you disconnect.
- It reduces operator error by putting the minimum required steps in one place and in the right order.
- It avoids the earlier transfer issues by giving a single recursive `scp` command for the dataset.
- It matches your intended workflow more closely because the documented training command now starts a new run rather than assuming a checkpoint already exists.

## Scope

These were documentation changes only. No training code or DigitalOcean scripts were modified in this pass.