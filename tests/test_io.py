import pandas as pd
import pytest
from pathlib import Path
from src.utils.io import load_csv


def test_load_csv_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_empty_file(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("")  # empty
    with pytest.raises(Exception):
        load_csv(p)
