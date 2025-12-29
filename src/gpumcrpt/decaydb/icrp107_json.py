from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json


@dataclass
class ICRP107Nuclide:
    name: str
    half_life_s: float
    emissions: Dict[str, Any]


_TIME_UNIT_TO_S = {
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "y": 365.25 * 86400.0,
}


def load_icrp107_json(path: str | Path) -> ICRP107Nuclide:
    """
    Load a single OpenGATE/ICRP107-style nuclide JSON record.

    Robustness:
    - Some provided files may be double-encoded JSON strings; handle that.
    """
    path = Path(path)
    txt = path.read_text(encoding="utf-8").strip()
    raw = json.loads(txt)
    if isinstance(raw, str):
        raw = json.loads(raw)

    name = str(raw["name"])
    hl = float(raw["half_life"])
    tu = str(raw["time_unit"])
    if tu not in _TIME_UNIT_TO_S:
        raise ValueError(f"Unknown time_unit={tu} in {path}")
    half_life_s = hl * _TIME_UNIT_TO_S[tu]
    emissions = dict(raw.get("emissions", {}))
    return ICRP107Nuclide(name=name, half_life_s=half_life_s, emissions=emissions)


def find_icrp107_json(db_dir: str | Path, nuclide_name: str) -> Path:
    """
    Locate <nuclide>.json in a directory.
    Accepts both 'Lu-177.json' and case-insensitive matches.
    """
    db_dir = Path(db_dir)
    if not db_dir.exists():
        raise FileNotFoundError(f"Decay DB directory not found: {db_dir}")

    candidates = [
        db_dir / f"{nuclide_name}.json",
        db_dir / f"{nuclide_name.upper()}.json",
        db_dir / f"{nuclide_name.lower()}.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: scan directory
    target = nuclide_name.lower()
    for p in db_dir.glob("*.json"):
        if p.stem.lower() == target:
            return p

    raise FileNotFoundError(f"Could not find {nuclide_name}.json in {db_dir}")


def load_icrp107_nuclide(db_dir: str | Path, nuclide_name: str) -> ICRP107Nuclide:
    return load_icrp107_json(find_icrp107_json(db_dir, nuclide_name))