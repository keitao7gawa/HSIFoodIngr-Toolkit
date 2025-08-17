from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py
import numpy as np


@dataclass
class CheckResult:
    status: str  # "ok" | "warn" | "error"
    messages: list[str]
    stats: Dict[str, Any]

    @property
    def exit_code(self) -> int:
        return {"ok": 0, "warn": 1, "error": 2}[self.status]


def _add_msg(res: CheckResult, level: str, msg: str) -> None:
    if level == "error":
        if res.status != "error":
            res.status = "error"
    elif level == "warn":
        if res.status == "ok":
            res.status = "warn"
    res.messages.append(f"{level.upper()}: {msg}")


def _exists(f: h5py.File, key: str, res: CheckResult) -> bool:
    if key not in f:
        _add_msg(res, "error", f"missing key: {key}")
        return False
    return True


def _safe_get(f: h5py.File, key: str, res: CheckResult):
    try:
        return f[key]
    except Exception as exc:
        _add_msg(res, "error", f"failed to access {key}: {exc}")
        return None


def check_file(path: str, strict: bool = False, samples: int = 8) -> CheckResult:
    res = CheckResult(status="ok", messages=[], stats={})
    try:
        with h5py.File(path, "r") as f:
            # existence
            for key in ("/hsi", "/rgb", "/masks", "/metadata"):
                if not _exists(f, key, res):
                    # cannot continue reliably without structure
                    continue

            if "/hsi" in f:
                hsi = _safe_get(f, "/hsi", res)
                if hsi is not None:
                    if hsi.ndim != 4:
                        _add_msg(res, "error", f"/hsi must be 4D (N,H,W,B), got {getattr(hsi,'shape',None)}")
                    if getattr(hsi, "dtype", None) != np.float32:
                        _add_msg(res, "warn" if not strict else "error", f"/hsi dtype expected float32, got {hsi.dtype}")
                    res.stats["hsi.shape"] = tuple(hsi.shape)

            if "/rgb" in f:
                rgb = _safe_get(f, "/rgb", res)
                if rgb is not None:
                    if rgb.ndim != 4 or rgb.shape[-1] != 3:
                        _add_msg(res, "error", f"/rgb must be 4D (N,H,W,3), got {getattr(rgb,'shape',None)}")
                    if getattr(rgb, "dtype", None) != np.uint8:
                        _add_msg(res, "warn" if not strict else "error", f"/rgb dtype expected uint8, got {rgb.dtype}")
                    res.stats["rgb.shape"] = tuple(rgb.shape)

            if "/masks" in f:
                masks = _safe_get(f, "/masks", res)
                if masks is not None:
                    if masks.ndim != 3:
                        _add_msg(res, "error", f"/masks must be 3D (N,H,W), got {getattr(masks,'shape',None)}")
                    res.stats["masks.shape"] = tuple(masks.shape)

            # metadata
            md = f["/metadata"] if "/metadata" in f else None
            if md is not None:
                # wavelengths
                if "wavelengths" in md:
                    wl = md["wavelengths"]
                    res.stats["wavelengths.len"] = int(wl.shape[0])
                    if "hsi" in res.stats:
                        pass
                # ingredient_map
                if "ingredient_map" in md:
                    try:
                        raw = md["ingredient_map"][()]
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        if isinstance(raw, np.ndarray) and raw.ndim >= 1:
                            raw = raw.reshape(-1)[0]
                            if isinstance(raw, bytes):
                                raw = raw.decode("utf-8")
                        json.loads(str(raw))
                    except Exception as exc:
                        _add_msg(res, "error", f"/metadata/ingredient_map invalid JSON: {exc}")

            # spot checks
            try:
                N = int(f["/hsi"].shape[0])
                res.stats["N"] = N
                import random

                choices = [0, N - 1]
                if N > 2:
                    choices.extend(random.sample(range(1, N - 1), k=min(samples, max(0, N - 2))))
                seen = set()
                for idx in choices:
                    if idx in seen:
                        continue
                    seen.add(idx)
                    hsi = np.asarray(f["/hsi"][idx], dtype=np.float32)  # (H,W,B)
                    rgb = np.asarray(f["/rgb"][idx])  # uint8
                    msk = np.asarray(f["/masks"][idx])
                    if not (rgb.dtype == np.uint8):
                        _add_msg(res, "warn" if not strict else "error", f"rgb[{idx}] dtype={rgb.dtype}")
                    if np.isnan(hsi).any() or np.isinf(hsi).any():
                        _add_msg(res, "error", f"hsi[{idx}] contains NaN/Inf")
                    if (rgb.min() < 0) or (rgb.max() > 255):
                        _add_msg(res, "error", f"rgb[{idx}] out of range [0,255]")
                    if msk.min() < 0:
                        _add_msg(res, "warn" if not strict else "error", f"masks[{idx}] has negative values")
            except Exception as exc:
                _add_msg(res, "error", f"spot check failed: {exc}")

    except OSError as exc:
        _add_msg(res, "error", f"failed to open file: {exc}")
    except Exception as exc:
        _add_msg(res, "error", f"unexpected error: {exc}")

    return res


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate HSIFoodIngr HDF5 file structure and contents")
    parser.add_argument("h5", type=str, help="Path to HDF5 file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors where applicable")
    parser.add_argument("--json", dest="json_path", type=str, default=None, help="Write JSON report to this path")
    parser.add_argument("--samples", type=int, default=8, help="Number of random samples for spot checks")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = check_file(args.h5, strict=args.strict, samples=args.samples)

    # human-readable output
    for m in result.messages:
        print(m)

    print(json.dumps({"status": result.status, **result.stats}, ensure_ascii=False))

    if args.json_path:
        try:
            with open(args.json_path, "w", encoding="utf-8") as fp:
                json.dump({
                    "status": result.status,
                    "messages": result.messages,
                    "stats": result.stats,
                }, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("failed to write JSON report: %s", exc)

    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
