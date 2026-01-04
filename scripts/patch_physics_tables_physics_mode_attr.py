from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass(frozen=True)
class PatchResult:
    path: Path
    old: str | None
    new: str | None
    changed: bool


def _decode_attr(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _canonical_from_filename(path: Path) -> str | None:
    name = path.name.lower()
    if "photon_electron_local" in name:
        return "photon_electron_local"
    if "photon_electron_condensed" in name:
        return "photon_electron_condensed"
    if "photon_em_energybucketed" in name:
        return "photon_em_energybucketed"
    if "local_deposit" in name:
        return "local_deposit"
    return None


def patch_file(path: Path) -> PatchResult:
    with h5py.File(path, "r+") as f:
        old = _decode_attr(f.attrs.get("physics_mode", None))
        target = _canonical_from_filename(path)
        if target is None:
            return PatchResult(path=path, old=old, new=old, changed=False)

        if old == target:
            return PatchResult(path=path, old=old, new=target, changed=False)

        f.attrs["physics_mode"] = target
        return PatchResult(path=path, old=old, new=target, changed=True)


def main() -> None:
    base = Path("src/gpumcrpt/physics_tables/precomputed_tables")
    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    h5_files = sorted(base.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {base}")

    results: list[PatchResult] = []
    for path in h5_files:
        results.append(patch_file(path))

    changed = [r for r in results if r.changed]

    for r in results:
        status = "CHANGED" if r.changed else "OK"
        print(f"{status}: {r.path}  physics_mode: {r.old!r} -> {r.new!r}")

    print(f"\nPatched {len(changed)} file(s) out of {len(results)}.")


if __name__ == "__main__":
    main()
