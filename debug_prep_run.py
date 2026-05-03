import sys
import traceback
from pathlib import Path

sys.path.insert(0, r'H:\Personal\Internships\WeKan\DBGDGM_Improvements')
from tools.wekan_prep.prepare_wekan_data import main

try:
    rc = main()
    Path(r'H:\Personal\Internships\WeKan\DBGDGM_Improvements\prep_rc.txt').write_text(str(rc), encoding='utf-8')
except Exception:
    Path(r'H:\Personal\Internships\WeKan\DBGDGM_Improvements\prep_exc.txt').write_text(traceback.format_exc(), encoding='utf-8')

