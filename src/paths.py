"""Path resolution independent of the directory from which Jupyter starts."""
from pathlib import Path


def get_project_root(current: Path | None = None) -> Path:
    location = (current or Path.cwd()).resolve()
    for candidate in [location, *location.parents]:
        if (candidate / 'src' / 'preprocessing.py').exists() and (candidate / 'notebooks').is_dir():
            return candidate
    raise FileNotFoundError('Start Jupyter within the repository or a subdirectory.')


def raw_data_path() -> Path:
    p = get_project_root() / 'data' / 'raw' / 'Train.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. Download the Kaggle dataset as described in README.md.')
    return p
