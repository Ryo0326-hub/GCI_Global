"""Validate the deliverable notebook without executing training or mounting Drive."""
import ast
import hashlib
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "comp.ipynb", as_version=4)
nbformat.validate(notebook)
pipeline = (root / "src/gci_pipeline.py").read_text()
embedded = []
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    source = cell.source
    if source.startswith("%%writefile gci_pipeline.py\n"):
        embedded.append(source.split("\n", 1)[1])
        source = embedded[-1]
    else:
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    ast.parse(source)
if embedded != [pipeline]:
    raise ValueError("Notebook and pipeline differ. Run scripts/build_notebook.py.")
if notebook.metadata.gci.pipeline_sha256 != hashlib.sha256(pipeline.encode()).hexdigest():
    raise ValueError("Notebook pipeline hash differs from its source.")
print(f"Validated {len(notebook.cells)} cells: schema, Python syntax and embedded pipeline.")
