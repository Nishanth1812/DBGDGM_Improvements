from pathlib import Path
import json
import sys

sys.path.insert(0, r'H:\Personal\Internships\WeKan\DBGDGM_Improvements')
from tools.wekan_prep.prepare_wekan_data import load_label_map, read_labels_from_csvs, resolve_subject_folder, build_subject_records

root = Path(r'C:\WeKan Training Data')
fmri_root = root / 'FMRI_DOWNLOAD_dataset'
smri_root = root / 'SMRI DOWNLOAD_dataset'
label_map = load_label_map(None)

fmri_labels = read_labels_from_csvs(fmri_root, label_map)
smri_labels = read_labels_from_csvs(smri_root, label_map)
records = build_subject_records(fmri_root, smri_root, label_map, allow_unknown_labels=True, accept_any_files=False)

payload = {
    'fmri_labels_count': len(fmri_labels),
    'smri_labels_count': len(smri_labels),
    'fmri_labels_sample': list(sorted(fmri_labels.items()))[:20],
    'smri_labels_sample': list(sorted(smri_labels.items()))[:20],
    'records_count': len(records),
    'records_sample': [
        {
            'subject_id': r.subject_id,
            'label': r.label,
            'fmri_src': str(r.fmri_src),
            'smri_src': str(r.smri_src),
        }
        for r in records[:20]
    ],
}

Path(r'H:\Personal\Internships\WeKan\DBGDGM_Improvements\probe_wekan.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print('wrote probe_wekan.json')

