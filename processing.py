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
  * red ground-mask check over the true-colour preview
  * AOI reading (KML / GeoJSON / zipped shapefile) + rasterisation
  * a folium web map (satellite basemap + index/class overlays + footprint + AOI)
  * index GeoTIFF and class shapefile (zipped) writers for download

Nothing here reads the multi-GB ortho; it all runs on the small precomputed
arrays, which is what keeps the hosted app light.
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
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
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
    return [plt.get_cmap("RdYlGn")(i / (n - 1)) for i in range(n)]


# --------------------------------------------------------------------------- #
#  Colour limits + rendering
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
    """RdYlGn RGBA uint8 image of an index array; off-mask pixels transparent.
    Returns a PIL image at the array's own resolution."""
    cmap = plt.get_cmap("RdYlGn")
    norm = np.clip((idx2d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgba = (cmap(norm) * 255).astype("uint8")
    rgba[~mask, 3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def colorize_classes(class_grid, n_classes):
    """RGBA uint8 image of the class grid (low->high palette); off-class (<0)
    transparent. Returns a PIL image at the array's own resolution."""
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


def fig_index_map(idx2d, mask, name, vmin, vmax, region="analysed area"):
    """Static matplotlib index map — used as a fallback when the web map can't
    render (e.g. folium unavailable or the dataset has no CRS/bounds)."""
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(alpha=0.0)
    disp = np.ma.masked_array(idx2d, mask=~mask)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{name} — {region}", fontsize=12)
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
    ax.set_title("Relative vigour classes", fontsize=12)
    ax.axis("off")
    labels = class_label_set(n_classes)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_classes)]
    ax.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def fig_histogram(values, edges, name):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(values, bins=60, color="#4575b4", alpha=0.85)
    for e in edges:
        ax.axvline(e, color="#d7191c", linestyle="--", linewidth=1)
    ax.set_xlabel(name)
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Distribution of {name} (dashed = class breaks)")
    fig.tight_layout()
    return fig


def fig_mask_histogram(mvals, otsu, threshold, name):
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.hist(mvals, bins=80, color="#a3babd")
    ax.axvline(otsu, color="#4575b4", linestyle=":", linewidth=1.6,
               label=f"Otsu auto ({otsu:.3f})")
    ax.axvline(threshold, color="#F46F29", linestyle="-", linewidth=2.0,
               label=f"Active threshold ({threshold:.3f})")
    ax.fill_betweenx([0, ax.get_ylim()[1]], ax.get_xlim()[0], threshold,
                     color="#d7191c", alpha=0.06)
    ax.set_xlabel(name)
    ax.set_ylabel("Pixel count")
    ax.set_title(f"{name} over all pixels — left of the line is masked as ground")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def _decode_rgb_png(rgb_png_bytes, shape_hw):
    """Decode the bundle's native-grid rgb.png to a float (H,W,3) array in 0..1.
    Resizes to match the index grid if needed. None if missing/undecodable."""
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


def fig_mask_check(bundle, valid, veg):
    """True-colour reference with pixels masked as ground tinted red, so the user
    can eyeball whether bare earth (not canopy) is being removed. Uses the bundle's
    native-grid rgb.png. (valid & ~veg) = removed-as-ground, shown red."""
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    axes[0].imshow(base)
    axes[0].set_title("True colour", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(over)
    axes[1].set_title("Masked as ground (red)", fontsize=12)
    axes[1].axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Area of interest (AOI): read a vector, rasterise to the display grid, and
#  reproject to WGS84 for drawing on the web map.
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
        gdf = gdf.set_crs(4326)                  # KML / loose files: assume WGS84
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
    """Reproject a raster-CRS shapely geom to EPSG:4326 and return a geojson mapping
    suitable for folium.GeoJson. None on failure."""
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
#  Web map (folium): satellite basemap + index/class overlays + footprint + AOI
# --------------------------------------------------------------------------- #
def build_map(bundle, idx, mask, index_name, vlo, vhi,
              class_grid=None, n_classes=None, aoi_geojson=None):
    """Return a folium.Map, or None if folium is unavailable or the dataset has no
    WGS84 bounds. Overlays are placed with a lat/lon image overlay — placement is
    approximate (fine for a scouting aid; exact georeferencing is in the downloads)."""
    bounds = bundle.get("bounds")
    if not bounds or "south" not in bounds:
        return None
    try:
        import folium
    except Exception:
        return None

    south, west = float(bounds["south"]), float(bounds["west"])
    north, east = float(bounds["north"]), float(bounds["east"])
    img_bounds = [[south, west], [north, east]]
    center = [(south + north) / 2.0, (west + east) / 2.0]

    m = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map",
                     overlay=False, control=True).add_to(m)

    # index overlay (shown by default)
    idx_img = colorize_index(idx, mask, vlo, vhi)
    folium.raster_layers.ImageOverlay(
        image=png_data_uri(idx_img), bounds=img_bounds, opacity=0.80,
        name=f"{index_name} index", interactive=False, zindex=1,
    ).add_to(m)

    # class overlay (hidden by default; toggle in the layer control)
    if class_grid is not None and n_classes:
        cls_img = colorize_classes(class_grid, n_classes)
        folium.raster_layers.ImageOverlay(
            image=png_data_uri(cls_img), bounds=img_bounds, opacity=0.80,
            name="Vigour classes", interactive=False, show=False, zindex=2,
        ).add_to(m)

    # ortho footprint
    folium.Rectangle(bounds=img_bounds, color="#1C2E36", weight=2,
                     fill=False, opacity=0.9).add_to(m)

    # AOI outline (if uploaded)
    if aoi_geojson is not None:
        folium.GeoJson(
            aoi_geojson, name="AOI",
            style_function=lambda _f: {"color": "#F46F29", "weight": 2.5,
                                       "fill": False},
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(img_bounds)
    return m


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


def classes_shapefile_zip(class_grid, n_classes, meta, sieve_min_pixels=8):
    """Polygonise the class raster and return a ZIPPED shapefile (bytes).

    Polygons carry the integer class (0..n-1) and its relative-vigour label. A
    small sieve drops specks below sieve_min_pixels so the file is usable in QGIS.
    Requires rasterio.features + geopandas; raises if geopandas is unavailable."""
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
    shp_path = os.path.join(tmp, "vigour_classes.shp")
    gdf.to_file(shp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(tmp):
            z.write(os.path.join(tmp, fn), arcname=fn)
    return buf.getvalue()
