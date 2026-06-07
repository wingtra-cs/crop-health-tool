"""
processing.py — analysis + rendering logic for the Crop Health web app.

Pure functions, independent of Streamlit and of storage. They operate on the
small display-resolution arrays that r2.load_bundle() returns (the indices, the
NDVI mask, and the validity mask), plus the bundle's affine transform / CRS for
the downloadable exports and the web map.

Crop-neutral port of the validated offline logic:
  * Otsu auto-threshold for separating bare ground from canopy
  * threshold application -> "vegetation" (canopy) mask
  * adaptive (per-block, percentile) colour limits so within-map variation shows
  * quartile / k-means relative-vigour classification (labels low -> high)
  * colourised index / class images for display and for the web map overlay
  * red ground-mask check over the true-colour preview (cropped to the analysed
    region / AOI so the subset fills the frame)
  * AOI reading (KML / GeoJSON / zipped shapefile) + rasterisation
  * a folium web map (satellite basemap with the ortho true-colour image as a
    persistent context layer; the index AND class overlays are reprojected to Web
    Mercator so they register correctly on the basemap, and are mutually exclusive
    via a grouped radio control so only one shows at a time; footprint + AOI + an
    on-map legend; fits to the AOI when one is given and allows deep zoom)
  * index GeoTIFF and class shapefile (zipped, with an AUTOMATIC speckle sieve
    scaled to the current map, and a named inner folder) writers for download

Charts use a clean "airy" style (white ground, faint gridlines, no top/right
frame, soft bars with an overlaid density curve, no titles, thin reference
lines). Nothing here reads the multi-GB ortho; it all runs on the small
precomputed arrays.
"""

import io
import os
import json
import glob
import base64
import zipfile
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb, to_hex
from PIL import Image

import rasterio
from rasterio import Affine
from rasterio.io import MemoryFile

try:
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# --------------------------------------------------------------------------- #
#  Index metadata (crop-neutral; describes what each index measures physically)
# --------------------------------------------------------------------------- #
INDEX_INFO = {
    "NDRE": {
        "label": "NDRE — Normalized Difference Red Edge",
        "formula": "(NIR - RedEdge) / (NIR + RedEdge)",
        "vmin": -0.1, "vmax": 0.8,
        "note": "Red-edge index sensitive to chlorophyll content and leaf density. "
                "Resists saturation in dense canopy, so it often separates vigour "
                "differences that NDVI flattens.",
    },
    "NDVI": {
        "label": "NDVI — Normalized Difference Vegetation Index",
        "formula": "(NIR - Red) / (NIR + Red)",
        "vmin": -0.1, "vmax": 0.9,
        "note": "Familiar greenness / canopy-cover proxy. Tends to saturate at high "
                "biomass; a good recognisable baseline.",
    },
    "CIre": {
        "label": "CIre — Red Edge Chlorophyll Index",
        "formula": "(NIR / RedEdge) - 1",
        "vmin": 0.0, "vmax": 3.0,
        "note": "Red-edge chlorophyll index, often more linear with chlorophyll at "
                "high biomass. Useful as a second layer alongside NDRE.",
    },
}

_INDEX_KEY = {"NDVI": "ndvi", "NDRE": "ndre", "CIre": "cire"}

CLASS_COLORS = ["#d7191c", "#fdae61", "#a6d96a", "#1a9641", "#006837"]
CLASS_LABELS = {
    3: ["Lowest", "Moderate", "Highest"],
    4: ["Lowest", "Low", "High", "Highest"],
    5: ["Lowest", "Low", "Moderate", "High", "Highest"],
}

# Gradient stops used for the index colour ramp legend (approximates RdYlGn).
_RAMP_STOPS = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]

# Automatic speckle sieve for the shapefile export: drop blobs smaller than
# SIEVE_FRACTION of the analysed (classified) area, but always at least
# SIEVE_FLOOR_PX pixels so trivial specks go even on tiny areas.
SIEVE_FRACTION = 0.0005   # 0.05% of the classified pixels
SIEVE_FLOOR_PX = 10

# Airy chart palette
_C_GRID = "#E6ECEE"
_C_SPINE = "#CBD5D8"
_C_LABEL = "#5b6b72"
_C_TITLE = "#1C2E36"
_C_BAR = "#9ec6e0"        # soft blue for the index histogram
_C_BAR2 = "#cdd8db"       # soft slate for the mask histogram
_C_CURVE = "#4a6b78"      # density-curve line (desaturated navy)
_C_OTSU = "#5b8fb0"       # muted blue dotted Otsu line
_C_THR = "#F46F29"        # Sun Orange for the active threshold (action accent)
_C_BREAK = "#cf5b4a"      # muted red dashed class breaks


def index_array(bundle, name):
    """Return the requested index array from a loaded bundle."""
    return bundle[_INDEX_KEY[name]]


# --------------------------------------------------------------------------- #
#  Thresholding
# --------------------------------------------------------------------------- #
def otsu_threshold(values, bins=256):
    """Otsu's method: the value that best separates two populations (bare ground
    vs canopy) in a 1-D set of values."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        return float(np.median(v))
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype("float64")
    mids = (edges[:-1] + edges[1:]) / 2.0
    wB = np.cumsum(hist)
    total = wB[-1]
    if total == 0:
        return float(np.median(v))
    wF = total - wB
    sum_cum = np.cumsum(hist * mids)
    sum_tot = sum_cum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mB = np.where(wB > 0, sum_cum / wB, 0.0)
        mF = np.where(wF > 0, (sum_tot - sum_cum) / wF, 0.0)
    between = wB * wF * (mB - mF) ** 2
    between[wF <= 0] = -1.0
    return float(mids[int(np.argmax(between))])


def slider_bounds(values):
    """Reasonable (lo, hi) bounds for a threshold slider over a value set."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.floor(np.nanpercentile(v, 1) * 100) / 100)
    hi = float(np.ceil(np.nanpercentile(v, 99) * 100) / 100)
    if hi <= lo:
        hi = lo + 0.1
    return lo, hi


def canopy_mask(bundle, ground_mask_on, mask_index_name, threshold, aoi_mask=None):
    """Boolean mask of pixels to analyse. Starts from validity (and AOI if given);
    if ground masking is on, keeps only pixels whose mask-index value >= threshold."""
    valid = bundle["valid"]
    if aoi_mask is not None:
        valid = valid & aoi_mask
    if not ground_mask_on:
        return valid
    m = bundle["mask_ndvi"] if mask_index_name == "NDVI" else index_array(bundle, mask_index_name)
    return valid & np.isfinite(m) & (m >= threshold)


# --------------------------------------------------------------------------- #
#  Classification
# --------------------------------------------------------------------------- #
def classify(values, method="quartile", n_classes=4):
    """Integer labels 0..n-1 (0 = lowest, n-1 = highest), plus bin edges/centroids."""
    values = np.asarray(values, dtype="float64")
    if method == "quartile":
        qs = np.linspace(0, 100, n_classes + 1)[1:-1]
        edges = np.percentile(values, qs)
        labels = np.digitize(values, edges)
        return labels.astype("int16"), edges
    elif method == "kmeans":
        if not HAVE_SKLEARN:
            raise RuntimeError("scikit-learn not installed; use the quartile method.")
        km = KMeans(n_clusters=n_classes, n_init=10, random_state=0)
        raw = km.fit_predict(values.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        order = np.argsort(centers)
        remap = np.zeros(n_classes, dtype="int16")
        remap[order] = np.arange(n_classes)
        return remap[raw].astype("int16"), np.sort(centers)
    raise ValueError(f"Unknown method {method}")


def class_label_set(n):
    return CLASS_LABELS.get(n, [f"Class {i + 1}" for i in range(n)])


def class_palette(n):
    if n <= 5:
        idx = np.linspace(0, 4, n).round().astype(int)
        return [CLASS_COLORS[i] for i in idx]
    return [to_hex(plt.get_cmap("RdYlGn")(i / (n - 1))) for i in range(n)]


# --------------------------------------------------------------------------- #
#  Colour limits + image rendering (for the map overlays)
# --------------------------------------------------------------------------- #
def adaptive_range(values, vmin_fallback, vmax_fallback, pclip=(2, 98)):
    """Percentile colour limits stretched to the data's own range so a narrow band
    of values still fills the ramp. Falls back to fixed limits if empty/constant."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(vmin_fallback), float(vmax_fallback)
    lo, hi = np.percentile(v, pclip)
    if hi - lo < 1e-6:
        lo, hi = lo - 0.05, hi + 0.05
    return float(lo), float(hi)


def colorize_index(idx2d, mask, vmin, vmax):
    """RdYlGn RGBA uint8 image of an index array. Alpha is set EXPLICITLY from the
    mask AND finiteness, so nodata / off-canopy pixels are fully transparent."""
    cmap = plt.get_cmap("RdYlGn")
    norm = (idx2d.astype("float64") - vmin) / max(vmax - vmin, 1e-6)
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype("uint8")
    keep = mask & np.isfinite(idx2d)
    rgba[..., 3] = np.where(keep, 255, 0).astype("uint8")
    rgba[~keep, :3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def colorize_classes(class_grid, n_classes):
    """RGBA uint8 image of the class grid (low->high palette); off-class (<0)
    transparent."""
    colors = class_palette(n_classes)
    rgba = np.zeros((*class_grid.shape, 4), dtype="uint8")
    for k in range(n_classes):
        r, g, b = to_rgb(colors[k])
        sel = class_grid == k
        rgba[sel, 0] = int(r * 255)
        rgba[sel, 1] = int(g * 255)
        rgba[sel, 2] = int(b * 255)
        rgba[sel, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def png_data_uri(img):
    """PIL image -> 'data:image/png;base64,...' string (for folium ImageOverlay)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
#  Charts — clean "airy" style: soft bars + overlaid density curve, no titles
# --------------------------------------------------------------------------- #
def _apply_airy(ax):
    """Shared airy treatment: white ground, faint y-gridlines behind data, no
    top/right frame, soft spines, muted ticks."""
    ax.set_facecolor("white")
    ax.grid(axis="y", color=_C_GRID, linewidth=0.9, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_C_SPINE)
    ax.tick_params(colors=_C_LABEL, labelsize=9, length=0)


def _density_xy(values, n=400):
    """KDE curve (xs, ys) over the data range, or None if scipy is unavailable or
    the data are degenerate. ys are scaled to a density (integrates to ~1)."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size < 5 or np.ptp(v) < 1e-9:
        return None
    try:
        from scipy.stats import gaussian_kde
        k = gaussian_kde(v)
        xs = np.linspace(v.min(), v.max(), n)
        return xs, k(xs)
    except Exception:
        return None


def fig_index_map(idx2d, mask, name, vmin, vmax, region="analysed area"):
    """Static matplotlib index map — fallback when the web map can't render."""
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(alpha=0.0)
    disp = np.ma.masked_array(idx2d, mask=~mask)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{name} — {region}", fontsize=12, color=_C_TITLE)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=name)
    fig.tight_layout()
    return fig


def fig_classified_map(class_grid, n_classes):
    """Static matplotlib classified map — fallback companion to fig_index_map."""
    colors = class_palette(n_classes)
    cmap = ListedColormap(colors)
    cmap.set_bad(alpha=0.0)
    disp = np.ma.masked_array(class_grid, mask=class_grid < 0)
    norm = BoundaryNorm(np.arange(-0.5, n_classes, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(disp, cmap=cmap, norm=norm)
    ax.set_title("Relative vigour classes", fontsize=12, color=_C_TITLE)
    ax.axis("off")
    labels = class_label_set(n_classes)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_classes)]
    ax.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def fig_histogram(values, edges, name):
    """Index distribution — airy bars + density curve, no title, thin lines."""
    fig, ax = plt.subplots(figsize=(8, 3.0))
    _apply_airy(ax)
    ax.hist(values, bins=60, color=_C_BAR, edgecolor="white", linewidth=0.4,
            alpha=0.55, density=True, zorder=3)
    dxy = _density_xy(values)
    if dxy is not None:
        ax.plot(dxy[0], dxy[1], color=_C_CURVE, linewidth=1.5, zorder=4)
    for e in edges:
        ax.axvline(e, color=_C_BREAK, linestyle=(0, (4, 3)), linewidth=0.8, zorder=5)
    ax.set_yticks([])
    ax.set_xlabel(name, color=_C_LABEL, fontsize=9)
    fig.tight_layout()
    return fig


def fig_mask_histogram(mvals, otsu, threshold, name):
    """Ground-threshold distribution — airy bars + density curve, no title, thin
    lines. Bars preserve the bimodal soil/canopy gap; the curve adds a clean line."""
    fig, ax = plt.subplots(figsize=(8, 3.0))
    _apply_airy(ax)
    ax.hist(mvals, bins=80, color=_C_BAR2, edgecolor="white", linewidth=0.3,
            alpha=0.5, density=True, zorder=3)
    dxy = _density_xy(mvals)
    if dxy is not None:
        ax.plot(dxy[0], dxy[1], color=_C_CURVE, linewidth=1.5, zorder=4)
    x0 = ax.get_xlim()[0]
    ax.axvspan(x0, threshold, color=_C_THR, alpha=0.05, zorder=1)
    ax.axvline(otsu, color=_C_OTSU, linestyle=":", linewidth=1.0, zorder=5,
               label=f"Otsu auto ({otsu:.3f})")
    ax.axvline(threshold, color=_C_THR, linewidth=1.3, zorder=6,
               label=f"Active threshold ({threshold:.3f})")
    ax.set_yticks([])
    ax.set_xlabel(name, color=_C_LABEL, fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _decode_rgb_png(rgb_png_bytes, shape_hw):
    """Decode the bundle's native-grid rgb.png to a float (H,W,3) array in 0..1."""
    if not rgb_png_bytes:
        return None
    try:
        im = Image.open(io.BytesIO(rgb_png_bytes)).convert("RGB")
        arr = np.asarray(im, dtype="float32") / 255.0
    except Exception:
        return None
    if arr.shape[0] != shape_hw[0] or arr.shape[1] != shape_hw[1]:
        try:
            im2 = Image.fromarray((arr * 255).astype("uint8")).resize(
                (shape_hw[1], shape_hw[0]))
            arr = np.asarray(im2, dtype="float32") / 255.0
        except Exception:
            return None
    return arr


def _crop_box(valid, margin_frac=0.06):
    """Row/col slice (r0, r1, c0, c1) bounding the True region of `valid`, padded
    by a margin so the subset doesn't touch the frame edge. Returns None if nothing
    is valid. Used to zoom the mask-check render onto the analysed region / AOI
    instead of showing the whole grid with the subset as a tiny patch."""
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not rows.any() or not cols.any():
        return None
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = int(r_idx[0]), int(r_idx[-1]) + 1
    c0, c1 = int(c_idx[0]), int(c_idx[-1]) + 1
    h, w = valid.shape
    mr = int(round((r1 - r0) * margin_frac)) + 1
    mc = int(round((c1 - c0) * margin_frac)) + 1
    return (max(0, r0 - mr), min(h, r1 + mr), max(0, c0 - mc), min(w, c1 + mc))


def fig_mask_check(bundle, valid, veg):
    """True-colour reference with pixels masked as ground tinted red. The render is
    cropped to the bounding box of `valid` (the analysed region, i.e. the AOI when
    one is set), so the subset fills the frame instead of sitting as a tiny patch
    in a large grey border."""
    shape_hw = valid.shape
    rgb = _decode_rgb_png(bundle.get("rgb_png"), shape_hw)
    if rgb is None:
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.text(0.5, 0.5, "No true-colour preview in this dataset for the mask check.",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        return fig

    bg = 0.5
    base = np.clip(rgb.copy(), 0, 1)
    base[~valid] = bg

    over = np.clip(rgb.copy(), 0, 1)
    over[~valid] = bg
    ground = valid & (~veg)
    red = np.array([0.86, 0.12, 0.12], dtype="float32")
    over[ground] = 0.45 * over[ground] + 0.55 * red

    # Zoom onto the analysed region / AOI.
    box = _crop_box(valid)
    if box is not None:
        r0, r1, c0, c1 = box
        base = base[r0:r1, c0:c1]
        over = over[r0:r1, c0:c1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    axes[0].imshow(base)
    axes[0].set_title("True colour", fontsize=12, color=_C_TITLE)
    axes[0].axis("off")
    axes[1].imshow(over)
    axes[1].set_title("Masked as ground (red)", fontsize=12, color=_C_TITLE)
    axes[1].axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Area of interest (AOI)
# --------------------------------------------------------------------------- #
def aoi_from_vector(name, data_bytes, raster_crs):
    """Read KML / GeoJSON / zipped-shapefile bytes, reproject to the raster CRS,
    and return a unioned shapely geometry."""
    try:
        import geopandas as gpd
        from shapely.ops import unary_union
    except Exception as e:
        raise RuntimeError(f"AOI reading needs geopandas/shapely: {e}")
    if raster_crs is None:
        raise ValueError("This dataset has no CRS, so a geographic AOI can't be placed.")

    nm = name.lower()
    tmp = tempfile.mkdtemp()
    if nm.endswith(".zip"):
        zp = os.path.join(tmp, "aoi.zip")
        with open(zp, "wb") as f:
            f.write(data_bytes)
        with zipfile.ZipFile(zp) as z:
            z.extractall(tmp)
        shps = glob.glob(os.path.join(tmp, "**", "*.shp"), recursive=True)
        if not shps:
            raise ValueError("No .shp file found inside the zip.")
        gdf = gpd.read_file(shps[0])
    else:
        fp = os.path.join(tmp, os.path.basename(nm) or "aoi")
        with open(fp, "wb") as f:
            f.write(data_bytes)
        gdf = gpd.read_file(fp)

    if gdf.empty:
        raise ValueError("No features found in the AOI file.")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(raster_crs)
    try:
        return gdf.geometry.union_all()
    except Exception:
        return unary_union(list(gdf.geometry))


def rasterize_aoi(geom, out_shape, transform):
    """Boolean mask (True inside the AOI) on the display grid, or None."""
    if geom is None:
        return None
    try:
        import rasterio.features
        from shapely.geometry import mapping
    except Exception:
        return None
    m = rasterio.features.rasterize(
        [(mapping(geom), 1)], out_shape=out_shape, transform=transform,
        fill=0, all_touched=True, dtype="uint8")
    return m.astype(bool)


def aoi_to_4326_geojson(geom, raster_crs):
    """Reproject a raster-CRS shapely geom to EPSG:4326; return a geojson mapping."""
    try:
        import geopandas as gpd
    except Exception:
        return None
    try:
        g = gpd.GeoSeries([geom], crs=raster_crs).to_crs(4326)
        return json.loads(g.to_json())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Web map (folium): satellite basemap + persistent true-colour ortho context
#  layer; index + classes reprojected to Web Mercator for correct registration,
#  loaded but mutually exclusive (radio) so only one shows at a time.
# --------------------------------------------------------------------------- #
def _mercator_overlay_grid(meta, shape_hw):
    """Compute one shared EPSG:3857 destination grid for the display arrays, so
    every overlay reprojects onto the same grid and stays co-registered. Returns
    (dst_transform, dst_w, dst_h, bounds_4326) or None if the dataset CRS is
    missing or rasterio.warp is unavailable.

    bounds_4326 = [[south, west], [north, east]] are the lat/lon corners of the
    3857 extent. Because Leaflet's CRS is Web Mercator, a folium ImageOverlay of a
    3857 image placed on those corners maps linearly with NO stretch — i.e. the
    overlay registers correctly on the satellite basemap (the previous plain
    lat/lon placement stretched the projected raster and was only approximate)."""
    src_transform = affine_from_meta(meta)
    src_crs = meta.get("crs")
    if src_transform is None or not src_crs:
        return None
    try:
        from rasterio.warp import calculate_default_transform, transform_bounds
        from rasterio.crs import CRS
    except Exception:
        return None
    try:
        src = CRS.from_user_input(src_crs)
        dst = CRS.from_epsg(3857)
        h, w = int(shape_hw[0]), int(shape_hw[1])
        left = src_transform.c
        top = src_transform.f
        right = left + src_transform.a * w
        bottom = top + src_transform.e * h
        dst_transform, dw, dh = calculate_default_transform(
            src, dst, w, h, left=left, bottom=bottom, right=right, top=top)
        d_left = dst_transform.c
        d_top = dst_transform.f
        d_right = d_left + dst_transform.a * dw
        d_bottom = d_top + dst_transform.e * dh
        w4, s4, e4, n4 = transform_bounds(dst, CRS.from_epsg(4326),
                                          d_left, d_bottom, d_right, d_top)
        return dst_transform, int(dw), int(dh), [[s4, w4], [n4, e4]]
    except Exception:
        return None


def _warp_rgba_to_3857(rgba, meta, dst_transform, dw, dh):
    """Reproject an (H,W,4) uint8 RGBA image from the dataset CRS to the shared
    EPSG:3857 grid (nearest-neighbour, per band — keeps class edges crisp and the
    alpha channel exact). Returns a PIL RGBA Image."""
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    src = CRS.from_user_input(meta.get("crs"))
    dst = CRS.from_epsg(3857)
    src_transform = affine_from_meta(meta)
    out = np.zeros((dh, dw, 4), dtype="uint8")
    for b in range(4):
        reproject(
            source=np.ascontiguousarray(rgba[..., b]),
            destination=out[..., b],
            src_transform=src_transform, src_crs=src,
            dst_transform=dst_transform, dst_crs=dst,
            resampling=Resampling.nearest)
    return Image.fromarray(out, mode="RGBA")


def _rgb_overlay_image(bundle):
    """RGBA true-colour image of the ortho for use as a PERSISTENT context layer on
    the web map: real pixels opaque, off-footprint transparent so the satellite
    basemap shows around it. Because it shares the overlays' grid it stays
    registered with the index/class layers, and being a real image it gives useful
    context even past the satellite's native zoom (where the basemap blurs).
    Returns None if the dataset has no true-colour preview."""
    valid = bundle.get("valid")
    if valid is None:
        return None
    rgb = _decode_rgb_png(bundle.get("rgb_png"), valid.shape)
    if rgb is None:
        return None
    rgba = np.zeros((*valid.shape, 4), dtype="uint8")
    rgba[..., :3] = np.clip(rgb * 255.0, 0, 255).astype("uint8")
    rgba[..., 3] = np.where(valid, 255, 0).astype("uint8")
    return Image.fromarray(rgba, mode="RGBA")


def _legend_html(index_name, vlo, vhi, n_classes):
    """Fixed legend panel showing BOTH the index gradient and the class swatches
    (both overlays exist on the map; the user toggles between them client-side)."""
    swatches = ""
    if n_classes:
        for c, lab in zip(class_palette(n_classes), class_label_set(n_classes)):
            hexc = c if isinstance(c, str) else to_hex(c)
            swatches += (
                f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="background:{hexc};width:13px;height:13px;display:inline-block;'
                f'margin-right:6px;border:1px solid #888;"></span>{lab}</div>')
    ramp = ", ".join(_RAMP_STOPS)
    return f"""
    <div style="position: fixed; bottom: 22px; right: 12px; z-index: 9999;
        background: rgba(255,255,255,0.93); padding: 9px 11px;
        border: 1px solid #bbb; border-radius: 6px; font-size: 12px;
        color: #1C2E36; font-family: sans-serif;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);">
      <div style="font-weight:700; margin-bottom:3px;">{index_name} index</div>
      <div style="background: linear-gradient(to right, {ramp});
          width:150px; height:12px; border:1px solid #888;"></div>
      <div style="display:flex; justify-content:space-between; width:150px;
          font-size:11px; margin-top:1px;">
        <span>{vlo:.2f}</span><span>low → high</span><span>{vhi:.2f}</span>
      </div>
      <div style="font-weight:700; margin:8px 0 3px;">Vigour classes</div>
      {swatches}
    </div>"""


def build_map(bundle, idx, mask, index_name, vlo, vhi,
              class_grid=None, n_classes=None, aoi_geojson=None):
    """Return a folium.Map, or None if folium is unavailable or the dataset has no
    WGS84 bounds. Satellite basemap, with the ortho true-colour image as a
    persistent context layer beneath the data (transparent off-footprint). The
    index and vigour-class overlays are REPROJECTED to Web Mercator so they
    register correctly on the basemap, then placed in one exclusive radio group so
    exactly one shows at a time (instant switch, no app reload). If reprojection
    isn't possible (no CRS / rasterio.warp missing), it falls back to the previous
    approximate lat/lon placement. Fits to the AOI when one is given; deep zoom is
    allowed for close inspection (the basemap blurs past native zoom, but the
    true-colour context and the overlays stay registered)."""
    bounds = bundle.get("bounds")
    if not bounds or "south" not in bounds:
        return None
    try:
        import folium
    except Exception:
        return None

    meta = bundle.get("meta") or {}
    south, west = float(bounds["south"]), float(bounds["west"])
    north, east = float(bounds["north"]), float(bounds["east"])
    foot_bounds = [[south, west], [north, east]]
    center = [(south + north) / 2.0, (west + east) / 2.0]

    # Build the colourised layers on the source grid.
    rgb_rgba = _rgb_overlay_image(bundle)
    idx_rgba = colorize_index(idx, mask, vlo, vhi)
    cls_rgba = (colorize_classes(class_grid, n_classes)
                if (class_grid is not None and n_classes) else None)

    # Reproject every layer onto one shared Web-Mercator grid so the linear
    # ImageOverlay placement is exact. All-or-nothing: if the warp fails we keep
    # the unwarped images on the footprint bounds (previous behaviour).
    overlay_bounds = foot_bounds
    grid = _mercator_overlay_grid(meta, idx.shape)
    if grid is not None:
        try:
            dst_t, dw, dh, b4326 = grid
            idx_w = _warp_rgba_to_3857(np.asarray(idx_rgba), meta, dst_t, dw, dh)
            cls_w = (_warp_rgba_to_3857(np.asarray(cls_rgba), meta, dst_t, dw, dh)
                     if cls_rgba is not None else None)
            rgb_w = (_warp_rgba_to_3857(np.asarray(rgb_rgba), meta, dst_t, dw, dh)
                     if rgb_rgba is not None else None)
            idx_rgba, cls_rgba, rgb_rgba = idx_w, cls_w, rgb_w
            overlay_bounds = b4326
        except Exception:
            overlay_bounds = foot_bounds  # unwarped images stay; original bounds

    m = folium.Map(location=center, zoom_start=15, tiles=None,
                   control_scale=True, max_zoom=22)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False, control=True,
        max_zoom=22, max_native_zoom=19,
    ).add_to(m)

    # Persistent true-colour ortho context layer, registered to the same bounds as
    # the data overlays and sitting just beneath them (zindex 0). Always on, not in
    # the layer control, so it stays behind whichever overlay is selected and keeps
    # giving context when the satellite blurs at deep zoom.
    if rgb_rgba is not None:
        folium.raster_layers.ImageOverlay(
            image=png_data_uri(rgb_rgba), bounds=overlay_bounds, opacity=1.0,
            name="True colour (ortho)", interactive=False, zindex=0,
            control=False, show=True).add_to(m)

    idx_layer = folium.raster_layers.ImageOverlay(
        image=png_data_uri(idx_rgba), bounds=overlay_bounds, opacity=0.85,
        name=f"{index_name} index", interactive=False, zindex=1, show=True)
    idx_layer.add_to(m)

    cls_layer = None
    if cls_rgba is not None:
        cls_layer = folium.raster_layers.ImageOverlay(
            image=png_data_uri(cls_rgba), bounds=overlay_bounds, opacity=0.85,
            name="Vigour classes", interactive=False, zindex=2, show=False)
        cls_layer.add_to(m)

    if aoi_geojson is not None:
        folium.Rectangle(bounds=overlay_bounds, color="#ffffff", weight=1.5,
                         fill=False, opacity=0.7, dash_array="6,6").add_to(m)
        folium.GeoJson(
            aoi_geojson, name="AOI",
            style_function=lambda _f: {"color": "#F46F29", "weight": 2.5,
                                       "fill": False},
        ).add_to(m)

    # Mutually exclusive overlays via a grouped radio control. If the installed
    # folium lacks GroupedLayerControl, fall back to a normal layer control.
    grouped = False
    if cls_layer is not None:
        try:
            from folium.plugins import GroupedLayerControl
            GroupedLayerControl(
                groups={"Layer": [idx_layer, cls_layer]},
                exclusive_groups=True, collapsed=False,
            ).add_to(m)
            grouped = True
        except Exception:
            grouped = False
    if not grouped:
        folium.LayerControl(collapsed=False).add_to(m)

    # Fit to the AOI when one is given, so the subset fills the view; otherwise fit
    # to the whole footprint.
    fit_target = overlay_bounds
    if aoi_geojson is not None:
        ab = _geojson_bounds(aoi_geojson)
        if ab is not None:
            fit_target = ab
    m.fit_bounds(fit_target)
    m.get_root().html.add_child(
        folium.Element(_legend_html(index_name, vlo, vhi, n_classes)))
    return m


def _geojson_bounds(gj):
    """Leaflet-style [[south, west], [north, east]] bounding box of a (EPSG:4326)
    geojson mapping, or None. Walks every coordinate pair in the features."""
    lons, lats = [], []

    def _walk(c):
        if (isinstance(c, (list, tuple)) and len(c) >= 2
                and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float))):
            lons.append(float(c[0]))
            lats.append(float(c[1]))
            return
        if isinstance(c, (list, tuple)):
            for x in c:
                _walk(x)

    try:
        feats = gj.get("features", []) if isinstance(gj, dict) else []
        for f in feats:
            geom = (f or {}).get("geometry") or {}
            _walk(geom.get("coordinates", []))
    except Exception:
        return None
    if not lons or not lats:
        return None
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


# --------------------------------------------------------------------------- #
#  Geo helpers + downloadable exports (built in memory)
# --------------------------------------------------------------------------- #
def affine_from_meta(meta):
    """rasterio Affine for the DISPLAY-res grid from the bundle meta transform."""
    t = meta.get("transform")
    if not t or len(t) < 6:
        return None
    return Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def pixel_area_m2(meta):
    if not meta.get("is_projected"):
        return None
    t = affine_from_meta(meta)
    return abs(t.a) * abs(t.e) if t else None


def index_geotiff_bytes(idx2d, mask, meta):
    """Single-band float32 GeoTIFF of the index over the mask (NaN elsewhere)."""
    data = np.where(mask, idx2d, np.nan).astype("float32")
    transform = affine_from_meta(meta)
    crs = meta.get("crs")
    h, w = data.shape
    profile = {"driver": "GTiff", "height": h, "width": w, "count": 1,
               "dtype": "float32", "nodata": np.nan, "compress": "deflate",
               "transform": transform, "crs": crs}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        return mem.read()


def auto_sieve_min_pixels(class_grid):
    """Automatic minimum blob size (pixels) for the shapefile sieve, scaled to the
    current map: SIEVE_FRACTION of the classified pixel count, with a floor so
    trivial specks are always removed. Returns 0 if there's nothing to classify."""
    classified = int((class_grid >= 0).sum())
    if classified <= 0:
        return 0
    return int(max(SIEVE_FLOOR_PX, round(SIEVE_FRACTION * classified)))


def classes_shapefile_zip(class_grid, n_classes, meta, folder_name="vigour_classes"):
    """Polygonise the class raster and return a ZIPPED shapefile (bytes).

    Speckle is removed AUTOMATICALLY: blobs smaller than a threshold derived from
    the current classified area (auto_sieve_min_pixels) are dropped — no user
    input, scales with the map. The shapefile components are written inside a named
    folder in the archive so it unzips cleanly to one directory. Polygons carry the
    integer class (0..n-1) and its relative-vigour label."""
    try:
        import geopandas as gpd
        from shapely.geometry import shape
        import rasterio.features
    except Exception as e:
        raise RuntimeError(f"Shapefile export needs geopandas/shapely: {e}")

    transform = affine_from_meta(meta)
    crs = meta.get("crs")
    grid = class_grid.astype("int32")
    valid = grid >= 0

    sieve_min_pixels = auto_sieve_min_pixels(grid)
    if sieve_min_pixels and sieve_min_pixels > 1:
        try:
            from scipy import ndimage as ndi
            for k in range(n_classes):
                mk = grid == k
                lab, n = ndi.label(mk)
                if n:
                    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
                    small = np.where(sizes < sieve_min_pixels)[0] + 1
                    if small.size:
                        grid[np.isin(lab, small) & mk] = -1
            valid = grid >= 0
        except Exception:
            pass

    labels = class_label_set(n_classes)
    geoms, recs = [], []
    for geom, val in rasterio.features.shapes(grid, mask=valid, transform=transform):
        k = int(val)
        if k < 0 or k >= n_classes:
            continue
        geoms.append(shape(geom))
        recs.append({"class": k, "label": labels[k]})
    if not geoms:
        raise RuntimeError("No class polygons to export.")

    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=crs)
    tmp = tempfile.mkdtemp()
    shp_path = os.path.join(tmp, f"{folder_name}.shp")
    gdf.to_file(shp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(tmp):
            # Nest every component inside a named folder for a clean unzip.
            z.write(os.path.join(tmp, fn), arcname=os.path.join(folder_name, fn))
    return buf.getvalue()
