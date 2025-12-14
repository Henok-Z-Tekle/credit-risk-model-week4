from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with basic validation.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame. Raises FileNotFoundError when the file is missing
        and ValueError when the file is empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV is empty: {p.resolve()}")
    return df
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with basic validation.

    Keeps the EDA notebook clean and reproducible.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV is empty: {p.resolve()}")
    return df
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with basic validation.

    Keeps the EDA notebook clean and reproducible.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV is empty: {p.resolve()}")
    return df
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with basic validation.

    Keeps the EDA notebook clean and reproducible.
    """
    p = Path(path)
    if not p.exists():
    raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p)
    if df.empty:
    raise ValueError(f"CSV is empty: {p.resolve()}")
    return df
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file with basic validation.
    Keeps the EDA notebook clean and reproducible.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV is empty: {p.resolve()}")
    return df
