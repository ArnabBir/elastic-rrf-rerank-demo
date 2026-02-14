import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Dict, Any, List, Union

from dotenv import find_dotenv, load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_env_path = find_dotenv(usecwd=True) or str(_PROJECT_ROOT / ".env")
load_dotenv(_env_path)
REPORTS_BASE = _PROJECT_ROOT / "reports"


def get_data_dir() -> Path:
    """Data directory for documents.jsonl, queries.jsonl, qrels.tsv. Set DATA_DIR env var to override."""
    path = os.getenv("DATA_DIR", "").strip()
    if path:
        p = Path(path)
        return (_PROJECT_ROOT / p) if not p.is_absolute() else p
    return _PROJECT_ROOT / "dataset" / "data_lite"


def get_report_dir(data_dir: Path) -> Path:
    """Report subdir per DATA_DIR: reports/{name}_{hash}/ for uniqueness."""
    resolved = str(data_dir.resolve())
    slug = hashlib.sha256(resolved.encode()).hexdigest()[:8]
    name = data_dir.name or "data"
    return REPORTS_BASE / f"{name}_{slug}"


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
