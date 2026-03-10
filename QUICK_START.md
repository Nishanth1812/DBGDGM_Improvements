# DBGDGM Preprocessing - Quick Navigation Guide

## Choose Your Path

### 🖥️ **Option 1: Local/Research (Preprocessing/)**

**Use if:**
- ✓ You have a local machine with 16+ GB RAM
- ✓ You want maximum control and customization
- ✓ You need advanced preprocessing options
- ✓ You're processing the full dataset
- ✓ You want to integrate with other tools

**Quick Start:**
```bash
cd Preprocessing/
python main.py --dataset both \
    --oasis-dir /path/to/oasis \
    --adni-dir /path/to/adni \
    --output-dir ./output \
    --verbose
```

**Key Files:**
- `Preprocessing/main.py` - Run this
- `Preprocessing/src/` - All processing logic
- `Preprocessing/config/preprocessing_config.yaml` - Customize here
- `Preprocessing/README.md` - Detailed instructions

---

### ☁️ **Option 2: Kaggle (Preprocessing_Kaggle/)**

**Use if:**
- ✓ You want to run directly on Kaggle
- ✓ Limited local compute/storage
- ✓ Prefer interactive notebook interface
- ✓ Want GPU acceleration from Kaggle
- ✓ Testing small datasets first

**Quick Start:**
1. Go to [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. Create new notebook
3. Copy content from:
   - `oasis_preprocessing_kaggle.ipynb` or
   - `adni_preprocessing_kaggle.ipynb`
4. Change dataset paths
5. Run all cells

**Key Files:**
- `Preprocessing_Kaggle/notebooks/oasis_preprocessing_kaggle.ipynb`
- `Preprocessing_Kaggle/notebooks/adni_preprocessing_kaggle.ipynb`
- `Preprocessing_Kaggle/README.md` - Kaggle-specific guide

---

## Decision Tree

```
START
  │
  ├─→ Do you have 16+ GB RAM? 
  │   YES → Use Preprocessing/ (Option 1)
  │   NO  → Go to next question
  │
  ├─→ Do you have Kaggle account?
  │   YES → Use Preprocessing_Kaggle/ (Option 2)
  │   NO  → Consider renting cloud GPU or use smaller dataset subset
  │
  └─→ Want to integrate with other tools?
      YES → Use Preprocessing/ (Option 1)
      NO  → Either works, Kaggle easier if you have account
```

## File Outputs (Both Versions Produce Same Format)

```
preprocessed_data/
├── OASIS/
│   ├── smri/
│   │   ├── OAS1_XXXX/MR1/
│   │   │   ├── features.npy        ← Input to DBGDGM
│   │   │   └── metadata.json
│   │   └── ...
│   └── metadata/subjects_summary.csv
│
├── ADNI/
│   ├── fmri/
│   │   ├── ADNI_XXX/BASELINE/
│   │   │   ├── timeseries_windows.npy  ← Input to DBGDGM
│   │   │   └── metadata.json
│   │   └── ...
│   ├── smri/
│   │   ├── ADNI_XXX/BASELINE/
│   │   │   ├── features.npy            ← Input to DBGDGM
│   │   │   └── metadata.json
│   │   └── ...
│   └── metadata/subjects_summary.csv
```

## Processing Times

### Local Version (Preprocessing/)
- OASIS-1 (416 subjects): ~30-40 mins
- ADNI baseline (1100+ subjects): ~2-3 hours
- Depends on CPU/disk speed

### Kaggle Version (Preprocessing_Kaggle/)
- OASIS-1 demo (5 subjects): ~2-3 mins
- ADNI demo (5 subjects): ~5-10 mins
- Full dataset: ~45 mins - 2 hours depending on size

## Example Workflows

### Workflow A: Quick Testing (Kaggle)
1. Upload 10-20 subjects to Kaggle
2. Run `oasis_preprocessing_kaggle.ipynb` or `adni_preprocessing_kaggle.ipynb`
3. Verify output shapes and values
4. Full dataset locally or in Kaggle dataset

### Workflow B: Local Processing (Recommended)
1. Download OASIS/ADNI to local drive
2. Configure `config/preprocessing_config.yaml`
3. Run: `python Preprocessing/main.py --dataset both ...`
4. Wait for completion
5. Load output into DBGDGM training

### Workflow C: Hybrid (Test + Process)
1. Run Kaggle version on subset
2. Verify preprocessing is correct
3. Run local version on full dataset

---

## Installation Requirements

### Local Version
```bash
cd Preprocessing/
pip install -r requirements.txt
```

### Kaggle Version
- Auto-installed in notebooks (no setup needed)

---

## Common Questions

**Q: Which version should I use for my thesis/publication?**  
A: Both produce identical outputs. Local version for reproducibility, Kaggle for speed.

**Q: Can I switch between versions?**  
A: Yes! Both produce same output format. Start with Kaggle to test, finish with local.

**Q: How do I handle multiple datasets?**  
A: Local version can process both. Kaggle: one notebook per dataset.

**Q: What if I run out of resources on Kaggle?**  
A: Process in batches or use local version. Kaggle has 9-hour runtime limit.

**Q: Can I modify preprocessing parameters?**  
A: Yes!
- Local: Edit `config/preprocessing_config.yaml`
- Kaggle: Modify inline in notebook cells

---

## Next Steps After Preprocessing

1. ✅ Choose and run your preprocessing version
2. ✅ Verify output shapes and quality
3. ✅ Load into PyTorch DataLoader
4. ✅ Train DBGDGM model from `Alzhiemers_Training/`
5. ✅ Analyze results

---

## Need Help?

- **Local setup issues?** → See `Preprocessing/README.md`
- **Kaggle issues?** → See `Preprocessing_Kaggle/README.md`
- **General guidance?** → See `PREPROCESSING_GUIDE.md`
- **Code details?** → Check inline documentation in files

---

**Ready? Pick your path above and get started!** 🚀
