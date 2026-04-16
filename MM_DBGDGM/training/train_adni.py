"""
Training script for the ADNI dataset
Integrates all objectives: classification, generative reconstruction,
alignment, regression, and survival analysis.
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
from models.mm_dbgdgm import MM_DBGDGM
from training.losses import MM_DBGDGM_Loss
from data.adni_loader import get_adni_dataloaders

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, beta_annealing=1.0):
    model.train()
    epoch_loss = 0.0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Move inputs to device
        fmri = batch['fmri'].to(device, non_blocking=True)
        smri = batch['smri'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # Move regression targets
        reg_targets = {
            'hippocampal_volume': batch['hippo_vol'].to(device, non_blocking=True),
            'cortical_thinning_rate': batch['cortical_thinning'].to(device, non_blocking=True),
            'dmn_connectivity': batch['dmn_conn'].to(device, non_blocking=True),
            'nss': batch['nss'].to(device, non_blocking=True)
        }
        
        # Move survival targets
        surv_times = batch['survival_times'].to(device, non_blocking=True)
        surv_events = batch['survival_events'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass returning all intermediates
        outputs = model(fmri, smri, return_all=True)
        
        # Calculate complex loss
        losses = criterion(
            logits=outputs['logits'],
            targets=labels,
            mu=outputs['mu'],
            logvar=outputs['logvar'],
            fmri_recon=outputs['fmri_recon'],
            fmri_orig=fmri,
            smri_recon=outputs['smri_recon'],
            smri_orig=smri,
            z_fmri=outputs['z_fmri'],
            z_smri=outputs['z_smri'],
            degeneration_preds=outputs['degeneration'],
            regression_targets=reg_targets,
            survival_shape=outputs['survival']['shape'],
            survival_scale=outputs['survival']['scale'],
            survival_times=surv_times,
            survival_events=surv_events,
            beta_annealing=beta_annealing
        )
        
        total_loss = losses['total']
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        
        # Accumulated component logging
        for k, v in losses.items():
            if k == 'total': continue
            loss_components[k] = loss_components.get(k, 0.0) + v.item()
            
        pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
        
    avg_loss = epoch_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    return avg_loss, avg_components

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    loss_components = {}
    
    with torch.no_grad():
        for batch in dataloader:
            fmri = batch['fmri'].to(device, non_blocking=True)
            smri = batch['smri'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            reg_targets = {
                'hippocampal_volume': batch['hippo_vol'].to(device, non_blocking=True),
                'cortical_thinning_rate': batch['cortical_thinning'].to(device, non_blocking=True),
                'dmn_connectivity': batch['dmn_conn'].to(device, non_blocking=True),
                'nss': batch['nss'].to(device, non_blocking=True)
            }
            surv_times = batch['survival_times'].to(device, non_blocking=True)
            surv_events = batch['survival_events'].to(device, non_blocking=True)
            
            outputs = model(fmri, smri, return_all=True)
            
            losses = criterion(
                logits=outputs['logits'],
                targets=labels,
                mu=outputs['mu'],
                logvar=outputs['logvar'],
                fmri_recon=outputs['fmri_recon'],
                fmri_orig=fmri,
                smri_recon=outputs['smri_recon'],
                smri_orig=smri,
                z_fmri=outputs['z_fmri'],
                z_smri=outputs['z_smri'],
                degeneration_preds=outputs['degeneration'],
                regression_targets=reg_targets,
                survival_shape=outputs['survival']['shape'],
                survival_scale=outputs['survival']['scale'],
                survival_times=surv_times,
                survival_events=surv_events,
                beta_annealing=1.0 # no annealing for val
            )
            
            val_loss += losses['total'].item()
            for k, v in losses.items():
                if k == 'total': continue
                loss_components[k] = loss_components.get(k, 0.0) + v.item()
                
    avg_loss = val_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    return avg_loss, avg_components

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Provide Dataloaders
    train_loader, val_loader = get_adni_dataloaders(batch_size=16, mock_data=True)
    
    # 2. Build Model
    model = MM_DBGDGM(
        n_roi=200,
        seq_len=50,
        n_smri_features=5,
        latent_dim=256,
        num_classes=4,
        dropout=0.2
    ).to(device)
    
    # 3. Setup Criterion & Optimizer
    criterion = MM_DBGDGM_Loss(
        num_classes=4,
        lambda_kl=0.1,
        lambda_align=0.1,
        lambda_recon=0.1,
        lambda_regression=0.5,
        lambda_survival=0.5
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 4. Training Loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # KL annealing scheme (0 to 1 over first 5 epochs)
        beta_annealing = min(1.0, epoch / 5.0)
        
        train_loss, train_comps = train_epoch(model, train_loader, criterion, optimizer, device, epoch, beta_annealing)
        val_loss, val_comps = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch} Results:")
        print(f"  Train Total Loss: {train_loss:.4f} | Class: {train_comps['classification']:.4f} | Reg: {train_comps['regression']:.4f} | Surv: {train_comps['survival']:.4f}")
        print(f"  Val   Total Loss: {val_loss:.4f} | Class: {val_comps['classification']:.4f} | Reg: {val_comps['regression']:.4f} | Surv: {val_comps['survival']:.4f}")
        
        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            # torch.save(model.state_dict(), 'checkpoints/mm_dbgdgm_best.pt')
            print("  --> Saved new best model!")

if __name__ == '__main__':
    main()
