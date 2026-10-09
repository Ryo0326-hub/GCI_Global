"""Import the notebook's report ZIP without overwriting edited experiment notes."""
import argparse
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
args = parser.parse_args()
count = 0
with zipfile.ZipFile(args.archive) as archive:
    for item in archive.infolist():
        if item.is_dir():
            continue
        relative = Path(item.filename)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".md" or relative.parts[:2] != ("Experiments", "Runs"):
            raise ValueError(f"Unexpected report path: {item.filename}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        contents = archive.read(item)
        if target.exists() and target.name != "Latest Run.md":
            if target.read_bytes() != contents:
                print(f"Kept existing edited note: {relative}")
            continue
        target.write_bytes(contents)
        count += 1
print(f"Imported {count} notes. Open 00 Dashboard in Obsidian.")
