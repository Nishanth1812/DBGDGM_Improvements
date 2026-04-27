import json
nb = json.load(open('notebooks/full_pipeline_demo.ipynb', encoding='utf-8'))
print(f'Valid JSON. {len(nb["cells"])} cells found.')
for i, c in enumerate(nb['cells']):
    print(f'  Cell {i}: {c["cell_type"]}')
