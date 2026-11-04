"""Prepare source-derived Python declarations and downloadable examples."""
from pathlib import Path
import shutil

from python_reference import prepare_reference

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parents[1]


def prepare_python():
    destination = DOCS / '_catalog/python'
    destination.mkdir(parents=True, exist_ok=True)
    prepare_reference(destination)
    downloads = DOCS / 'public/downloads/examples'
    downloads.mkdir(parents=True, exist_ok=True)
    for name in ['dataloader-example.py', 'processfn-example.py', 'loader-example.py']:
        shutil.copy2(ROOT / 'example' / name, downloads / name)
    print('Python documentation: source-derived declarations and 3 example downloads')


if __name__ == '__main__':
    prepare_python()
