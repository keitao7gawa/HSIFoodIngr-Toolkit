from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import streamlit as st
except Exception as _streamlit_exc:  # pragma: no cover
    st = None  # type: ignore

try:
    from hsifoodingr.datasets import HSIFoodIngrDataset
except Exception as exc:  # pragma: no cover
    HSIFoodIngrDataset = None  # type: ignore
    _dataset_import_error = exc
else:
    _dataset_import_error = None


@dataclass
class GUIArgs:
    h5: Optional[str] = None
    sample: Optional[int] = None
    pseudo: str = "rgb"  # rgb|band|pseudocolor
    band: Optional[int] = None
    save_dir: Optional[str] = None


def parse_cli_args(argv: Optional[List[str]] = None) -> GUIArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--h5", type=str, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--pseudo", type=str, default="rgb")
    parser.add_argument("--band", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    try:
        ns, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        # In Streamlit context, ignore parse errors
        ns = parser.parse_args([])
    return GUIArgs(h5=ns.h5, sample=ns.sample, pseudo=ns.pseudo, band=ns.band, save_dir=ns.save_dir)


# ---- small utilities ----

def _nearest_wavelength_indices(wavelengths: Optional[np.ndarray], targets_nm: Sequence[float]) -> List[int]:
    if wavelengths is None or len(wavelengths) == 0:
        # fallback: spread indices
        return [int(t) for t in targets_nm]
    idxs: List[int] = []
    for t in targets_nm:
        diffs = np.abs(wavelengths - float(t))
        idxs.append(int(np.argmin(diffs)))
    return idxs


def _minmax_scale(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min + eps)).astype(np.float32)


def _hsi_band_to_rgb(hsi_chw: np.ndarray, band_index: int) -> np.ndarray:
    # hsi_chw: (B,H,W) -> return (H,W,3) grayscale repeated
    b, h, w = hsi_chw.shape
    band_index = max(0, min(b - 1, int(band_index)))
    g = _minmax_scale(hsi_chw[band_index])
    img = np.stack([g, g, g], axis=-1)
    return img


def _pseudocolor_from_wavelengths(
    hsi_chw: np.ndarray,
    wavelengths: Optional[np.ndarray],
    rgb_targets_nm: Tuple[float, float, float] = (650.0, 550.0, 450.0),
) -> np.ndarray:
    # Map nearest wavelength bands to R,G,B and scale each to [0,1]
    if wavelengths is not None and wavelengths.size == hsi_chw.shape[0]:
        r_idx, g_idx, b_idx = _nearest_wavelength_indices(wavelengths, rgb_targets_nm)
    else:
        # fallback: evenly spaced bands
        b = hsi_chw.shape[0]
        r_idx, g_idx, b_idx = max(0, b // 4 * 3 - 1), max(0, b // 2 - 1), max(0, b // 4 - 1)
    r = _minmax_scale(hsi_chw[r_idx])
    g = _minmax_scale(hsi_chw[g_idx])
    b_ = _minmax_scale(hsi_chw[b_idx])
    rgb = np.stack([r, g, b_], axis=-1)
    return rgb


def _parse_color(value: Any, default_seed: int) -> Tuple[int, int, int, str]:
    # Accept '#RGB' or '#RRGGBB' or [r,g,b] or {'r':..,'g':..,'b':..}
    if isinstance(value, str) and value.startswith("#") and len(value) in (4, 7):
        if len(value) == 4:
            r = int(value[1] * 2, 16)
            g = int(value[2] * 2, 16)
            b = int(value[3] * 2, 16)
        else:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
        return r, g, b, value
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        r, g, b = [int(max(0, min(255, int(v)))) for v in value[:3]]
        return r, g, b, f"#{r:02x}{g:02x}{b:02x}"
    if isinstance(value, dict):
        if all(k in value for k in ("r", "g", "b")):
            r, g, b = [int(max(0, min(255, int(value[k])))) for k in ("r", "g", "b")]
            return r, g, b, f"#{r:02x}{g:02x}{b:02x}"
        if "hex" in value and isinstance(value["hex"], str):
            return _parse_color(value["hex"], default_seed)
    # fallback deterministic color
    rng = np.random.default_rng(default_seed)
    r, g, b = [int(x) for x in (rng.integers(64, 255), rng.integers(64, 255), rng.integers(64, 255))]
    return r, g, b, f"#{r:02x}{g:02x}{b:02x}"


def _extract_id_name_color_map(ingredient_map: Optional[Dict[str, Any]]) -> Dict[int, Tuple[str, Optional[Any]]]:
    """Return mapping: id -> (name, color_raw_or_None) from various schemas.

    Accepts schemas like:
      - {"classes": [{"id": 1, "name": "Rice", "color": "#aabbcc"}, ...]}
      - {"0": {"name": "Rice", "color": [255,128,0]}, ...}
      - {"id_to_name": {"0": "Rice", ...}, "id_to_color": {"0": "#..."}}
      - {"0": "Rice", "1": "..."}
    """
    mapping: Dict[int, Tuple[str, Optional[Any]]] = {}
    if not isinstance(ingredient_map, dict):
        return mapping

    # Case 0: name -> id mapping (e.g., {"background":0, "Rice":1, ...})
    try:
        if len(ingredient_map) > 0 and all(isinstance(v, (int, np.integer, str)) for v in ingredient_map.values()):
            inverted: Dict[int, Tuple[str, Optional[Any]]] = {}
            for name, cid in ingredient_map.items():
                try:
                    cid_int = int(cid)
                except Exception:
                    continue
                inverted[cid_int] = (str(name), None)
            if len(inverted) > 0:
                mapping.update(inverted)
                return mapping
    except Exception:
        pass

    # Case 1: explicit classes list
    classes = ingredient_map.get("classes") if isinstance(ingredient_map, dict) else None
    if isinstance(classes, list):
        for c in classes:
            try:
                cid = int(c.get("id", c.get("index", c.get("class_id"))))
                # prefer common name keys
                name = (
                    c.get("name")
                    or c.get("label")
                    or c.get("ja")
                    or c.get("en")
                    or str(cid)
                )
                color = c.get("color", c.get("hex", c.get("rgb")))
                mapping[cid] = (str(name), color)
            except Exception:
                continue

    # Case 2: id_to_* dicts
    id_to_name = ingredient_map.get("id_to_name") if isinstance(ingredient_map, dict) else None
    id_to_color = ingredient_map.get("id_to_color") if isinstance(ingredient_map, dict) else None
    if isinstance(id_to_name, dict):
        for k, v in id_to_name.items():
            try:
                cid = int(k)
                color = id_to_color.get(k) if isinstance(id_to_color, dict) else None
                mapping.setdefault(cid, (str(v), color))
            except Exception:
                continue

    # Case 3: flat dict keyed by id
    for k, v in ingredient_map.items():
        try:
            cid = int(k)
        except Exception:
            continue
        if cid in mapping:
            continue
        if isinstance(v, dict):
            name = v.get("name") or v.get("label") or v.get("ja") or v.get("en") or str(cid)
            color = v.get("color", v.get("hex", v.get("rgb")))
        else:
            name = str(v)
            color = None
        mapping[cid] = (str(name), color)

    return mapping


def _mask_to_color(mask_hw: np.ndarray, ingredient_map: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, str, str]]]:
    # Returns (rgba uint8 image HxWx4, legend)
    h, w = mask_hw.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    id_to_info: Dict[int, Tuple[int, int, int, str, str]] = {}

    # Build color and name map from metadata (preferred)
    meta_map = _extract_id_name_color_map(ingredient_map)
    for cid, (name, color_raw) in meta_map.items():
        r, g, b, hexcol = _parse_color(color_raw, default_seed=cid)
        id_to_info[int(cid)] = (r, g, b, name, hexcol)

    # Default colormap for unseen ids or missing color
    def color_for_id(cid: int) -> Tuple[int, int, int, str, str]:
        if cid in id_to_info:
            r, g, b, name, hexcol = id_to_info[cid]
            return r, g, b, name, hexcol
        # If name is known but color not set, use deterministic fallback
        name = meta_map.get(cid, (str(cid), None))[0]
        r, g, b, hexcol = _parse_color(None, default_seed=cid)
        info = (r, g, b, name, hexcol)
        id_to_info[cid] = info
        return info

    # Paint RGBA
    unique_ids = np.unique(mask_hw)
    for cid in unique_ids:
        r, g, b, _name, _hex = color_for_id(int(cid))
        sel = mask_hw == cid
        rgba[sel, 0] = r
        rgba[sel, 1] = g
        rgba[sel, 2] = b
        rgba[sel, 3] = 255

    return rgba, id_to_info


# ---- Streamlit App ----

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    try:
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) == 3:
            hex_color = "".join([c * 2 for c in hex_color])
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return r, g, b
    except Exception:
        return 255, 0, 255  # magenta fallback


def _draw_marker(
    img: np.ndarray,
    x: int,
    y: int,
    color: Tuple[int, int, int] = (255, 0, 255),
    size: int = 7,
    thickness: int = 2,
    crosshair: bool = True,
) -> np.ndarray:
    """Overlay a crosshair marker on the given HxWx3 uint8 image and return a copy."""
    out = img.copy()
    if out.ndim != 3 or out.shape[2] != 3:
        return out
    h, w, _ = out.shape
    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))
    size = max(1, int(size))
    thickness = max(1, int(thickness))

    # Horizontal line
    x1 = max(0, x - size)
    x2 = min(w - 1, x + size)
    y1 = max(0, y - (thickness // 2))
    y2 = min(h - 1, y + (thickness - 1) // 2)
    out[y1 : y2 + 1, x1 : x2 + 1, 0] = color[0]
    out[y1 : y2 + 1, x1 : x2 + 1, 1] = color[1]
    out[y1 : y2 + 1, x1 : x2 + 1, 2] = color[2]

    # Vertical line
    y1v = max(0, y - size)
    y2v = min(h - 1, y + size)
    x1v = max(0, x - (thickness // 2))
    x2v = min(w - 1, x + (thickness - 1) // 2)
    out[y1v : y2v + 1, x1v : x2v + 1, 0] = color[0]
    out[y1v : y2v + 1, x1v : x2v + 1, 1] = color[1]
    out[y1v : y2v + 1, x1v : x2v + 1, 2] = color[2]

    return out

def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    if st is None:
        raise RuntimeError("streamlit is required to run this app")
    if HSIFoodIngrDataset is None:
        st.error(f"Failed to import dataset: {_dataset_import_error}\nInstall torch, h5py, numpy, pillow, streamlit.")
        st.stop()

    args = parse_cli_args(argv)

    st.set_page_config(page_title="HSIFoodIngr Visualizer", layout="wide")
    st.title("HSIFoodIngr Visualizer")

    # Sidebar: file/path
    st.sidebar.header("Data Source")
    h5_path = st.sidebar.text_input("HDF5 path", value=args.h5 or "", help="Path to HSIFoodIngr-64.h5")
    normalize = st.sidebar.selectbox("HSI normalization", options=["none", "minmax", "standard"], index=1)
    return_uint8_rgb = st.sidebar.checkbox("Return uint8 RGB (0..255)", value=False)

    # Load dataset lazily and cache
    @st.cache_resource(show_spinner=True)
    def _load_dataset(path: str, normalize: str, uint8_rgb: bool) -> HSIFoodIngrDataset:
        return HSIFoodIngrDataset(path, normalize=normalize, return_uint8_rgb=uint8_rgb)

    ds: Optional[HSIFoodIngrDataset] = None
    if h5_path:
        try:
            ds = _load_dataset(h5_path, normalize, return_uint8_rgb)
        except Exception as exc:
            st.error(f"Failed to open HDF5: {exc}")
            ds = None

    if ds is None:
        st.info("Set a valid HDF5 path to start.")
        st.stop()

    N = len(ds)
    st.sidebar.write(f"Samples: {N}")

    # Retrieve one sample to know shapes
    sample_index = st.sidebar.number_input("Sample index", min_value=0, max_value=max(0, N - 1), value=min(args.sample or 0, max(0, N - 1)), step=1)
    sample = ds[int(sample_index)]

    hsi = sample["hsi"].numpy()  # (B,H,W) float32
    rgb = sample["rgb"].numpy()  # (3,H,W) float32 or uint8
    mask = sample["mask"].numpy()  # (H,W)
    meta = sample.get("meta", {})
    wavelengths = meta.get("wavelengths", None)
    if isinstance(wavelengths, np.ndarray):
        wavelengths = wavelengths.astype(np.float32)
    ingredient_map = meta.get("ingredient_map", None)

    B, H, W = hsi.shape

    # Display options
    st.sidebar.header("Display")
    mode = st.sidebar.selectbox("Mode", options=["RGB", "HSI band", "Pseudocolor"], index={"rgb":0,"band":1,"pseudocolor":2}.get(args.pseudo,0))
    show_mask = st.sidebar.checkbox("Show mask overlay", value=True)
    alpha = st.sidebar.slider("Mask alpha", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

    if mode == "RGB":
        if rgb.dtype != np.uint8:
            rgb_img = np.clip((rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8), 0, 255)
        else:
            rgb_img = rgb.transpose(1, 2, 0)
        base_img = rgb_img.astype(np.uint8)
    elif mode == "HSI band":
        band_index = st.sidebar.slider("Band index", min_value=0, max_value=B - 1, value=min(args.band or (B // 2), B - 1), step=1)
        base_img = (_hsi_band_to_rgb(hsi, band_index) * 255.0).astype(np.uint8)
    else:  # Pseudocolor
        base_img = (_pseudocolor_from_wavelengths(hsi, wavelengths) * 255.0).astype(np.uint8)

    # Mask overlay
    if show_mask:
        mask_rgba, legend = _mask_to_color(mask, ingredient_map)
        # alpha blend
        out = base_img.copy().astype(np.float32)
        a = (mask_rgba[..., 3:4].astype(np.float32) / 255.0) * float(alpha)
        out = (1.0 - a) * out + a * mask_rgba[..., :3].astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.uint8)
        vis_img = out
    else:
        vis_img = base_img

    # Spectral profile: choose coordinates
    st.sidebar.header("Spectral profile")
    x = st.sidebar.slider("x", min_value=0, max_value=W - 1, value=W // 2, step=1)
    y = st.sidebar.slider("y", min_value=0, max_value=H - 1, value=H // 2, step=1)

    # Marker options
    st.sidebar.header("Marker")
    show_marker = st.sidebar.checkbox("Show marker on image", value=True)
    marker_size = st.sidebar.slider("Marker size (px)", min_value=3, max_value=21, value=9, step=1)
    marker_color_hex = st.sidebar.color_picker("Marker color", value="#ff00ff")
    marker_color = _hex_to_rgb(marker_color_hex)

    # Layout: image and profile
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Visualization")
        disp_img = vis_img
        if show_marker:
            disp_img = _draw_marker(disp_img, int(x), int(y), color=marker_color, size=int(marker_size), thickness=2)
        st.image(disp_img, caption=f"Sample {int(sample_index)} ({mode})", use_container_width=True)
    with col2:
        st.subheader("Spectral profile")
        spec = hsi[:, int(y), int(x)]  # (B,)
        if isinstance(wavelengths, np.ndarray) and wavelengths.shape[0] == B:
            # Line with wavelengths
            import altair as alt
            df = {
                "wavelength": wavelengths.tolist(),
                "value": spec.tolist(),
            }
            chart = alt.Chart(alt.Data(values=[{"wavelength": float(w), "value": float(v)} for w, v in zip(df["wavelength"], df["value"])])).mark_line().encode(
                x=alt.X("wavelength:Q", title="Wavelength (nm)"),
                y=alt.Y("value:Q", title="Intensity"),
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            # Fallback: index axis
            import altair as alt
            df = [{"band": int(i), "value": float(v)} for i, v in enumerate(spec)]
            chart = alt.Chart(alt.Data(values=df)).mark_line().encode(
                x=alt.X("band:Q", title="Band index"),
                y=alt.Y("value:Q", title="Intensity"),
            )
            st.altair_chart(chart, use_container_width=True)

        # Legend
        st.subheader("Legend")
        if ingredient_map is not None:
            _, legend_map = _mask_to_color(mask, ingredient_map)
            present_ids = np.unique(mask)
            # Build HTML table with color squares
            rows_html = []
            for cid in sorted(int(c) for c in present_ids.tolist()):
                info = legend_map.get(cid)
                if info is None:
                    continue
                r, g, b, name, hexcol = info
                swatch = f'<span style="display:inline-block;width:0.9em;height:0.9em;background-color:{hexcol};border:1px solid #999;vertical-align:middle;margin-right:0.4em;"></span>'
                color_cell = f"{swatch}{hexcol}"
                rows_html.append(f"<tr><td style='padding:4px 8px;'>{cid}</td><td style='padding:4px 8px;'>{name}</td><td style='padding:4px 8px;'>{color_cell}</td></tr>")
            html = (
                "<table style='border-collapse:collapse;'>"
                "<thead><tr><th style='text-align:left;padding:4px 8px;'>id</th><th style='text-align:left;padding:4px 8px;'>name</th><th style='text-align:left;padding:4px 8px;'>color</th></tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody>"
                "</table>"
            )
            if rows_html:
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.write("No labels present in this mask.")
        else:
            st.write("No ingredient_map available.")

    # Download current view
    st.sidebar.header("Export")
    buf = io.BytesIO()
    Image.fromarray(vis_img).save(buf, format="PNG")
    st.sidebar.download_button("Download PNG", data=buf.getvalue(), file_name=f"sample{int(sample_index)}_{mode.lower()}.png", mime="image/png")


if __name__ == "__main__":  # pragma: no cover
    main()
