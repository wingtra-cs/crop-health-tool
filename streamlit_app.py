"""
streamlit_app.py — Crop Health Visualization and Classification Tool

Hosted web front-end (Streamlit Community Cloud).

This version wires processing.py into the UI: branding + password gate, an
ortho dropdown driven live by the R2 bucket, then the interactive analysis —
bare-earth threshold (Otsu + slider), vegetation-index map (adaptive colour
scale), relative-vigour classification (quartile / k-means, 3–5 classes), a
per-class table, and three downloads (full ortho via presigned URL; the selected
index as GeoTIFF; the classes as a zipped shapefile).

The map overlay (Leaflet) and AOI subsetting are added in the next step. Storage
access lives in r2.py; all analysis/rendering lives in processing.py.

Secrets (Community Cloud "Secrets" box, or local .streamlit/secrets.toml):

    app_password = "..."
    [r2]
    account_id = "..."
    access_key = "..."   # READ-ONLY token
    secret_key = "..."
    bucket     = "ptpn-bucket"

Branding: optional logo at assets/wingtra_logo.png (committed; not a secret).
"""

import io
import os
import hmac

import numpy as np
import streamlit as st

import r2
import processing as proc


APP_TITLE = "Crop Health Visualization and Classification Tool"
LOGO_PATH = "assets/wingtra_logo.png"


# --------------------------------------------------------------------------- #
#  Branding
# --------------------------------------------------------------------------- #
def render_brand(show_title=True):
    has_logo = os.path.exists(LOGO_PATH)
    if has_logo:
        try:
            st.logo(LOGO_PATH)
        except Exception:
            pass
    if not show_title:
        return
    if has_logo:
        c1, c2 = st.columns([1, 6], vertical_alignment="center")
        with c1:
            st.image(LOGO_PATH, width=110)
        with c2:
            st.title(APP_TITLE)
    else:
        st.title(APP_TITLE)


# --------------------------------------------------------------------------- #
#  Password gate (soft gate; real protection is private bucket + presigned TTL)
# --------------------------------------------------------------------------- #
def check_password():
    if st.session_state.get("auth_ok"):
        return True
    render_brand()
    st.caption("Enter the access password to continue.")
    pw = st.text_input("Password", type="password", key="pw_input")
    if pw:
        expected = st.secrets.get("app_password", "")
        if expected and hmac.compare_digest(pw, expected):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# --------------------------------------------------------------------------- #
#  Per-ortho threshold state: re-seed Otsu when the dataset / mask index changes
# --------------------------------------------------------------------------- #
def setup_threshold(bundle, mask_index_name):
    """Compute the mask-index values, Otsu (memoised per dataset+index), and the
    slider bounds; initialise the slider value once per signature. Returns
    (mvals, otsu, lo, hi)."""
    valid = bundle["valid"]
    mask_full = bundle["mask_ndvi"] if mask_index_name == "NDVI" \
        else proc.index_array(bundle, mask_index_name)
    mvals = mask_full[valid & np.isfinite(mask_full)]
    if mvals.size == 0:
        return mvals, None, None, None

    sig = (bundle["slug"], mask_index_name)
    if st.session_state.get("otsu_sig") != sig:
        st.session_state["otsu_sig"] = sig
        st.session_state["otsu_val"] = proc.otsu_threshold(mvals)
    otsu = st.session_state["otsu_val"]

    lo, hi = proc.slider_bounds(mvals)
    tsig = (bundle["slug"], mask_index_name, lo, hi)
    if st.session_state.get("thr_sig") != tsig:
        st.session_state["thr_sig"] = tsig
        st.session_state["mask_thr"] = float(min(max(round(otsu, 3), lo), hi))
    return mvals, otsu, lo, hi


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_password():
        st.stop()

    render_brand()

    # ---- Sidebar: dataset + options ---------------------------------- #
    with st.sidebar:
        st.header("Dataset")
        if st.button("↻ Refresh list", help="Re-check storage for datasets"):
            r2.refresh_orthos()
            st.rerun()

    try:
        orthos = r2.list_orthos()
    except Exception as e:
        st.error("Could not reach storage. Check the app's R2 credentials.")
        st.caption(f"Details: {e}")
        st.stop()
    if not orthos:
        st.info("No datasets available yet. Upload an orthomosaic bundle to "
                "storage and it will appear here automatically.")
        st.stop()

    names = [o["name"] for o in orthos]
    with st.sidebar:
        picked = st.selectbox("Orthomosaic", names, index=0)
        st.divider()

        st.header("Index")
        index_name = st.selectbox(
            "Vegetation index", list(proc.INDEX_INFO.keys()),
            format_func=lambda k: proc.INDEX_INFO[k]["label"])
        st.caption(f"**{index_name}** = {proc.INDEX_INFO[index_name]['formula']}")
        st.caption(proc.INDEX_INFO[index_name]["note"])

        st.header("Ground mask")
        ground_mask_on = st.checkbox(
            "Mask out bare ground", value=True,
            help="Separate canopy from soil / roads / gaps using a threshold found "
                 "automatically (Otsu) and adjustable below.")
        mask_index_name = "NDVI"
        if ground_mask_on:
            use_ndvi = st.checkbox("Use NDVI for the ground mask (recommended)",
                                   value=True)
            mask_index_name = "NDVI" if use_ndvi else index_name

        st.header("Classification")
        methods = ["quartile"] + (["kmeans"] if proc.HAVE_SKLEARN else [])
        method = st.selectbox("Method", methods,
                              help="Quartile = equal-count bins. "
                                   "K-means = natural breaks.")
        n_classes = st.slider("Number of classes", 3, 5, 4)

    selected = orthos[names.index(picked)]
    slug = selected["slug"]

    # ---- Load bundle ------------------------------------------------- #
    try:
        with st.spinner("Loading dataset…"):
            bundle = r2.load_bundle(slug)
    except FileNotFoundError as e:
        st.error("This dataset looks incomplete in storage and can't be loaded.")
        st.caption(f"Details: {e}")
        st.stop()
    except Exception as e:
        st.error("Could not load this dataset.")
        st.caption(f"Details: {e}")
        st.stop()

    meta = bundle["meta"]
    info = proc.INDEX_INFO[index_name]
    px_area = proc.pixel_area_m2(meta)

    st.subheader(meta.get("name", slug))

    # ---- Ground threshold (Otsu + slider) ---------------------------- #
    threshold = None
    if ground_mask_on:
        mvals, otsu, lo, hi = setup_threshold(bundle, mask_index_name)
        if otsu is None:
            st.warning("No valid pixels to threshold in this dataset.")
            st.stop()

        with st.expander("Ground mask threshold", expanded=True):
            cthr, cbtn = st.columns([3, 1])
            with cthr:
                threshold = st.slider(
                    f"Threshold on {mask_index_name} (below = bare ground)",
                    min_value=lo, max_value=hi, step=0.005, key="mask_thr")
            with cbtn:
                st.metric("Otsu auto", f"{otsu:.3f}")
                if st.button("Reset to Otsu"):
                    st.session_state["mask_thr"] = float(min(max(round(otsu, 3),
                                                                 lo), hi))
                    st.rerun()
            st.pyplot(proc.fig_mask_histogram(mvals, otsu, threshold,
                                              mask_index_name))

    # ---- Build the analysis mask + index values --------------------- #
    idx = proc.index_array(bundle, index_name)
    mask = proc.canopy_mask(bundle, ground_mask_on, mask_index_name, threshold)
    vals = idx[mask & np.isfinite(idx)]
    if vals.size == 0:
        st.warning("No pixels to analyse — lower the ground threshold or turn the "
                   "ground mask off.")
        st.stop()

    region = "vegetated canopy" if ground_mask_on else "analysed area"
    vlo, vhi = proc.adaptive_range(vals, info["vmin"], info["vmax"])

    # ---- Index map --------------------------------------------------- #
    st.markdown(f"### {index_name} map")
    st.pyplot(proc.fig_index_map(idx, mask, index_name, vlo, vhi, region=region))
    st.caption(f"Colour scale stretched to this dataset's {index_name} range "
               f"({vlo:.2f}–{vhi:.2f}, 2–98th percentile) to bring out relative "
               "variation — qualitative and within-map only, not comparable "
               "between datasets.")

    # ---- Statistics -------------------------------------------------- #
    st.markdown("### Statistics")
    count = int(mask.sum())
    sc = st.columns(4)
    sc[0].metric("Analysed pixels", f"{count:,}")
    if px_area:
        sc[1].metric("Analysed area", f"{count * px_area / 1e4:,.2f} ha")
    else:
        sc[1].metric("Analysed area", "n/a (geographic CRS)")
    sc[2].metric(f"Mean {index_name}", f"{np.nanmean(vals):.3f}")
    sc[3].metric(f"Median {index_name}", f"{np.nanmedian(vals):.3f}")

    # ---- Classification ---------------------------------------------- #
    st.markdown("### Relative vigour classification")
    try:
        labels, edges = proc.classify(vals, method=method, n_classes=n_classes)
    except Exception as e:
        st.error(str(e))
        st.stop()

    class_grid = np.full(idx.shape, -1, dtype="int16")
    class_grid[mask & np.isfinite(idx)] = labels

    cc = st.columns([3, 2])
    with cc[0]:
        st.pyplot(proc.fig_classified_map(class_grid, n_classes))
    with cc[1]:
        st.pyplot(proc.fig_histogram(vals, edges, index_name))

    label_names = proc.class_label_set(n_classes)
    rows = []
    for k in range(n_classes):
        cnt = int((labels == k).sum())
        pct = 100.0 * cnt / labels.size
        area = f"{cnt * px_area / 1e4:,.2f}" if px_area else "n/a"
        rows.append({"Class": label_names[k], "Pixels": f"{cnt:,}",
                     "% of area": f"{pct:.1f}%", "Area (ha)": area})
    st.table(rows)
    st.caption("Classes are relative bands within *this* dataset — a pixel's rank "
               "in the index distribution, not a health diagnosis. 'Lowest' marks "
               "where to look first on the ground. Low values can also reflect "
               "normal phenology (e.g. seasonal leaf fall), not necessarily a problem.")

    # ---- Downloads --------------------------------------------------- #
    st.markdown("### Downloads")
    d1, d2, d3 = st.columns(3)

    # (1) full ortho — presigned direct-from-storage link
    with d1:
        try:
            url = r2.ortho_download_url(bundle, ttl=900)
            st.link_button("⬇ 5-band orthomosaic", url, use_container_width=True)
            st.caption("Direct from storage; link ~15 min.")
        except Exception as e:
            st.caption(f"Ortho link unavailable: {e}")

    # (2) index GeoTIFF — generated from the display-res array
    with d2:
        try:
            tif = proc.index_geotiff_bytes(idx, mask, meta)
            st.download_button(f"⬇ {index_name} GeoTIFF", data=tif,
                               file_name=f"{slug}_{index_name}.tif",
                               mime="image/tiff", use_container_width=True)
            st.caption("Display-resolution, georeferenced.")
        except Exception as e:
            st.caption(f"Index export unavailable: {e}")

    # (3) classes as a zipped shapefile
    with d3:
        if st.button("Prepare class shapefile", use_container_width=True):
            try:
                with st.spinner("Polygonising classes…"):
                    zbytes = proc.classes_shapefile_zip(class_grid, n_classes, meta)
                st.session_state["shp_bytes"] = zbytes
                st.session_state["shp_name"] = f"{slug}_{index_name}_classes.zip"
            except Exception as e:
                st.session_state.pop("shp_bytes", None)
                st.error(f"Shapefile export failed: {e}")
        if st.session_state.get("shp_bytes"):
            st.download_button("⬇ Class shapefile (zip)",
                               data=st.session_state["shp_bytes"],
                               file_name=st.session_state.get("shp_name",
                                                              "classes.zip"),
                               mime="application/zip", use_container_width=True)
            st.caption("Polygonised, sieved for QGIS.")

    st.divider()
    st.caption("Map overlay and AOI subsetting are added next. Analysis runs on a "
               f"~{meta.get('display_shape',[0,0])[0]}px working copy; downloads of "
               "the index/classes are at that resolution, the ortho download is full.")


if __name__ == "__main__":
    main()
