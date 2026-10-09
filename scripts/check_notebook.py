"""Validate the deliverable notebook without executing training or mounting Drive."""
import ast
import hashlib
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "comp.ipynb", as_version=4)
nbformat.validate(notebook)
pipeline = (root / "src/gci_pipeline.py").read_text()
environment = (root / "src/gci_environment.py").read_text()
pins = dict(line.split("==") for line in (root / "requirements.txt").read_text().splitlines() if line)
embedded = []
embedded_environments = []
embedded_pins = []
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    source = cell.source
    if source.startswith("%%writefile gci_pipeline.py\n"):
        embedded.append(source.split("\n", 1)[1])
        source = embedded[-1]
    else:
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "environment_file" and node.func.attr == "write_text":
            embedded_environments.append(ast.literal_eval(node.args[0]))
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "gci_environment" and node.func.attr == "prepare_colab":
            embedded_pins.append(ast.literal_eval(node.args[0]))
if embedded != [pipeline]:
    raise ValueError("Notebook and pipeline differ. Run scripts/build_notebook.py.")
if notebook.metadata.gci.pipeline_sha256 != hashlib.sha256(pipeline.encode()).hexdigest():
    raise ValueError("Notebook pipeline hash differs from its source.")
if embedded_environments != [environment] or embedded_pins != [pins]:
    raise ValueError("Notebook environment setup differs from its source or requirements. Run scripts/build_notebook.py.")
if notebook.metadata.gci.environment_sha256 != hashlib.sha256(environment.encode()).hexdigest():
    raise ValueError("Notebook environment hash differs from its source.")
print(f"Validated {len(notebook.cells)} cells: schema, syntax, embedded pipeline, setup and dependency pins.")
