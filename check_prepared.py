from pathlib import Path
root = Path(r'C:\WeKan Training Data\mm_dbgdgm_prepared')
lines = [f'root_exists={root.exists()}', f'labels_exists={(root / "labels.csv").exists() if root.exists() else False}']
if root.exists():
    lines.append('top_level=' + ';'.join(p.name for p in root.iterdir()))
    fmri_root = root / 'fmri'
    smri_root = root / 'smri'
    lines.append('fmri_dirs=' + (str(sum(1 for p in fmri_root.iterdir() if p.is_dir())) if fmri_root.exists() else '0'))
    lines.append('smri_dirs=' + (str(sum(1 for p in smri_root.iterdir() if p.is_dir())) if smri_root.exists() else '0'))
    lines.append('file_count=' + str(sum(1 for _ in root.rglob('*') if _.is_file())))
Path(r'H:\Personal\Internships\WeKan\DBGDGM_Improvements\prep_summary_py.txt').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))

