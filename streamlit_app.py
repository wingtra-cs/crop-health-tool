"""
streamlit_app.py — Crop Health Visualization and Classification Tool

Hosted web front-end (Streamlit Community Cloud).

Branding + password gate, an ortho dropdown driven live by the R2 bucket (no
dataset loads until the user actively picks one), then the interactive analysis:
  * bare-earth threshold (Otsu + slider) with a red ground-mask check
  * optional AOI subsetting (upload KML / GeoJSON / zipped shapefile)
  * a satellite web map with the index and the vigour classes as MUTUALLY
    EXCLUSIVE overlays (radio in the map's layer control — only one shows, and
    switching is instant with no app reload), plus an on-map legend + AOI outline
  * relative-vigour classification (quartile / k-means, 3–5 classes) + table
  * three downloads: full ortho (presigned URL), index GeoTIFF, class shapefile
    (speckle is sieved out automatically, scaled to the current map; the prepared
    file is invalidated whenever any defining setting changes, so a download is
    always for the current settings)

Visual style: clean / minimal — Manrope type, a Mercury-blue header band with a
Sun-Orange rule, white rounded "cards" around the graphics, airy charts.

Storage access lives in r2.py; all analysis/rendering lives in processing.py.

Secrets (Community Cloud "Secrets" box, or local .streamlit/secrets.toml):

    app_password = "..."
    [r2]
    account_id = "..."
    access_key = "..."   # READ-ONLY token
    secret_key = "..."
    bucket     = "ptpn-bucket"

Branding: optional logo at assets/wingtra_logo.png (committed; not a secret).
"""

import os
import hmac
import base64

import numpy as np
import streamlit as st

import r2
import processing as proc

try:
    from streamlit_folium import st_folium
    HAVE_FOLIUM = True
except Exception:
    HAVE_FOLIUM = False


APP_TITLE = "Crop Health Visualization and Classification Tool"
LOGO_PATH = "assets/wingtra_logo.png"
PLACEHOLDER = "— Select a dataset —"

MERCURY = "#1C2E36"
SUN_ORANGE = "#F46F29"
URANUS = "#A3BABD"
CANVAS = "#EDF2F2"
CARD_BORDER = "#E2E8EA"

_APP_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

  html, body, .stApp, [class*="css"], button, input, select, textarea {{
      font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif !important;
  }}
  .stApp {{ background-color: {CANVAS}; }}

  .ch-band {{
      background: {MERCURY}; border-bottom: 3px solid {SUN_ORANGE};
      padding: 16px 26px; margin: -1.2rem -1.2rem 20px -1.2rem;
      display: flex; align-items: center; gap: 16px;
  }}
  .ch-band img {{ height: 30px; }}
  .ch-band h1 {{
      color: #FFFFFF; font-size: 22px; font-weight: 800;
      margin: 0; line-height: 1.2; letter-spacing: .2px;
  }}
  .ch-band .ch-sub {{ color: {URANUS}; font-size: 12.5px; font-weight: 500; }}

  div[data-testid="stMetric"] {{
      background: #FFFFFF; border: 1px solid {CARD_BORDER};
      border-radius: 12px; padding: 12px 16px;
      box-shadow: 0 1px 3px rgba(28,46,54,0.06);
  }}
  div[data-testid="stMetricLabel"] p {{ color: #6b7b82; font-weight: 600; }}

  div[data-testid="stVerticalBlockBorderWrapper"] {{
      background: #FFFFFF; border: 1px solid {CARD_BORDER} !important;
      border-radius: 14px !important;
      box-shadow: 0 1px 4px rgba(28,46,54,0.07);
  }}

  h3 {{ color: {MERCURY}; font-weight: 700; letter-spacing: .2px; }}
  .stDownloadButton button, .stLinkButton a {{ border-radius: 10px; }}
  section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {CARD_BORDER}; }}
</style>
"""


# --------------------------------------------------------------------------- #
#  Branding
# --------------------------------------------------------------------------- #
def _logo_b64():
    if os.path.exists(LOGO_PATH):
        try:
            return base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
        except Exception:
            return None
    return None


def render_header(subtitle="Multispectral canopy analysis"):
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    b64 = _logo_b64()
    brand = (f'<img src="data:image/png;base64,{b64}" alt="Wingtra"/>' if b64
             else '<span style="color:#F46F29;font-weight:800;font-size:22px;">wingtra</span>')
    sub = f'<div class="ch-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ch-band">{brand}<div><h1>{APP_TITLE}</h1>{sub}</div></div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Password gate
# --------------------------------------------------------------------------- #
def check_password():
    if st.session_state.get("auth_ok"):
        return True
    render_header()
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
#  Threshold helpers
# --------------------------------------------------------------------------- #
def _reset_threshold(value):
    st.session_state["mask_thr"] = value


def setup_threshold(bundle, mask_index_name, aoi_mask=None):
    valid = bundle["valid"]
    if aoi_mask is not None:
        valid = valid & aoi_mask
    mask_full = bundle["mask_ndvi"] if mask_index_name == "NDVI" \
        else proc.index_array(bundle, mask_index_name)
    mvals = mask_full[valid & np.isfinite(mask_full)]
    if mvals.size == 0:
        return mvals, None, None, None

    aoi_key = int(aoi_mask.sum()) if aoi_mask is not None else 0
    sig = (bundle["slug"], mask_index_name, aoi_key)
    if st.session_state.get("otsu_sig") != sig:
        st.session_state["otsu_sig"] = sig
        st.session_state["otsu_val"] = proc.otsu_threshold(mvals)
    otsu = st.session_state["otsu_val"]

    lo, hi = proc.slider_bounds(mvals)
    seed = float(min(max(round(otsu, 3), lo), hi))
    tsig = (bundle["slug"], mask_index_name, aoi_key, lo, hi)
    if st.session_state.get("thr_sig") != tsig or "mask_thr" not in st.session_state:
        st.session_state["thr_sig"] = tsig
        st.session_state["mask_thr"] = seed
    return mvals, otsu, lo, hi


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_password():
        st.stop()

    render_header()

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
        picked = st.selectbox("Orthomosaic", [PLACEHOLDER] + names, index=0)
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

        st.header("Area of interest (optional)")
        aoi_file = st.file_uploader(
            "Restrict to an AOI — KML, GeoJSON, or zipped shapefile",
            type=["kml", "geojson", "json", "zip"],
            help="Clip the analysis to a sub-area. The AOI is drawn on the map for "
                 "context. Leave empty to analyse the whole orthomosaic.")

    if picked == PLACEHOLDER:
        st.info("Select a dataset from the sidebar to begin.")
        st.stop()

    selected = orthos[names.index(picked)]
    slug = selected["slug"]

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

    # ---- AOI (optional) ---------------------------------------------- #
    aoi_mask = None
    aoi_geojson = None
    if aoi_file is not None:
        try:
            geom = proc.aoi_from_vector(aoi_file.name, aoi_file.getvalue(),
                                        meta.get("crs"))
            transform = proc.affine_from_meta(meta)
            shape_hw = tuple(meta.get("display_shape", bundle["valid"].shape))
            m_ = proc.rasterize_aoi(geom, shape_hw, transform)
            if m_ is not None and m_.sum() > 0:
                aoi_mask = m_
                aoi_geojson = proc.aoi_to_4326_geojson(geom, meta.get("crs"))
                ha = (f"{int(aoi_mask.sum()) * px_area / 1e4:,.2f} ha"
                      if px_area else f"{int(aoi_mask.sum()):,} px")
                st.caption(f"AOI applied — analysis restricted to {ha}.")
            else:
                st.warning("The AOI doesn't overlap this orthomosaic — analysing the "
                           "whole dataset instead.")
        except Exception as e:
            st.error(f"Could not read the AOI: {e}")

    # ---- Ground threshold + red mask check --------------------------- #
    threshold = None
    if ground_mask_on:
        mvals, otsu, lo, hi = setup_threshold(bundle, mask_index_name, aoi_mask)
        if otsu is None:
            st.warning("No valid pixels to threshold in this dataset / AOI.")
            st.stop()

        with st.expander("Ground mask threshold", expanded=True):
            cthr, cbtn = st.columns([3, 1])
            with cthr:
                threshold = st.slider(
                    f"Threshold on {mask_index_name} (below = bare ground)",
                    min_value=lo, max_value=hi, step=0.005, key="mask_thr")
            with cbtn:
                st.metric("Otsu auto", f"{otsu:.3f}")
                st.button("Reset to Otsu", on_click=_reset_threshold,
                          args=(float(min(max(round(otsu, 3), lo), hi)),))

            st.pyplot(proc.fig_mask_histogram(mvals, otsu, threshold,
                                              mask_index_name))

            valid_eff = bundle["valid"] & aoi_mask if aoi_mask is not None \
                else bundle["valid"]
            veg_preview = proc.canopy_mask(bundle, True, mask_index_name,
                                           threshold, aoi_mask=aoi_mask)
            st.pyplot(proc.fig_mask_check(bundle, valid_eff, veg_preview))
            st.caption("Red marks pixels removed as bare ground at the current "
                       "threshold. Raise it if red covers canopy; lower it if soil / "
                       "roads / gaps aren't caught.")

    # ---- Analysis mask + index values -------------------------------- #
    idx = proc.index_array(bundle, index_name)
    mask = proc.canopy_mask(bundle, ground_mask_on, mask_index_name, threshold,
                            aoi_mask=aoi_mask)
    vals = idx[mask & np.isfinite(idx)]
    if vals.size == 0:
        st.warning("No pixels to analyse — lower the ground threshold, clear the AOI, "
                   "or turn the ground mask off.")
        st.stop()

    region = "vegetated canopy" if ground_mask_on else "analysed area"
    vlo, vhi = proc.adaptive_range(vals, info["vmin"], info["vmax"])

    # ---- Classification ---------------------------------------------- #
    try:
        labels, edges = proc.classify(vals, method=method, n_classes=n_classes)
    except Exception as e:
        st.error(str(e))
        st.stop()
    class_grid = np.full(idx.shape, -1, dtype="int16")
    class_grid[mask & np.isfinite(idx)] = labels

    # ---- Map card (index + classes as mutually exclusive overlays) --- #
    st.markdown("### Map")
    with st.container(border=True):
        fmap = None
        if HAVE_FOLIUM:
            try:
                fmap = proc.build_map(bundle, idx, mask, index_name, vlo, vhi,
                                      class_grid=class_grid, n_classes=n_classes,
                                      aoi_geojson=aoi_geojson)
            except Exception as e:
                st.caption(f"Map unavailable ({e}); showing static maps.")
                fmap = None
        if fmap is not None:
            st_folium(fmap, height=560, returned_objects=[],
                      use_container_width=True)
            st.caption("Satellite basemap. Switch between the index and the vigour "
                       "classes in the layer control (top right) — only one shows at "
                       "a time. Overlay placement is approximate; the georeferenced "
                       "products are in the downloads.")
        else:
            st.pyplot(proc.fig_index_map(idx, mask, index_name, vlo, vhi,
                                         region=region))
            st.pyplot(proc.fig_classified_map(class_grid, n_classes))
            st.caption("Static fallback (map renderer unavailable).")

    # ---- Statistics (stat cards) ------------------------------------- #
    st.markdown("### Statistics")
    count = int(mask.sum())
    sc = st.columns(4)
    sc[0].metric("Analysed pixels", f"{count:,}")
    if px_area:
        sc[1].metric("Analysed area", f"{count * px_area / 1e4:,.2f} ha")
    else:
        sc[1].metric("Analysed area", "n/a")
    sc[2].metric(f"Mean {index_name}", f"{np.nanmean(vals):.3f}")
    sc[3].metric(f"Median {index_name}", f"{np.nanmedian(vals):.3f}")

    # ---- Classification card ----------------------------------------- #
    st.markdown("### Relative vigour classification")
    with st.container(border=True):
        cc = st.columns([3, 2])
        with cc[0]:
            label_names = proc.class_label_set(n_classes)
            rows = []
            for k in range(n_classes):
                cnt = int((labels == k).sum())
                pct = 100.0 * cnt / labels.size
                area = f"{cnt * px_area / 1e4:,.2f}" if px_area else "n/a"
                rows.append({"Class": label_names[k], "Pixels": f"{cnt:,}",
                             "% of area": f"{pct:.1f}%", "Area (ha)": area})
            st.table(rows)
        with cc[1]:
            st.pyplot(proc.fig_histogram(vals, edges, index_name))
        st.caption("Classes are relative bands within *this* dataset — a pixel's rank "
                   "in the index distribution, not a health diagnosis. 'Lowest' marks "
                   "where to look first on the ground. Low values can also reflect "
                   "normal phenology (e.g. seasonal leaf fall), not necessarily a problem.")

    # ---- Downloads card ---------------------------------------------- #
    st.markdown("### Downloads")
    with st.container(border=True):
        d1, d2, d3 = st.columns(3)
        with d1:
            try:
                url = r2.ortho_download_url(bundle, ttl=900)
                st.link_button("⬇ 5-band orthomosaic", url, use_container_width=True)
                st.caption("Direct from storage; link ~15 min.")
            except Exception as e:
                st.caption(f"Ortho link unavailable: {e}")
        with d2:
            try:
                tif = proc.index_geotiff_bytes(idx, mask, meta)
                st.download_button(f"⬇ {index_name} GeoTIFF", data=tif,
                                   file_name=f"{slug}_{index_name}.tif",
                                   mime="image/tiff", use_container_width=True)
                st.caption("Display-resolution, georeferenced.")
            except Exception as e:
                st.caption(f"Index export unavailable: {e}")
        with d3:
            # Signature of everything that defines the prepared shapefile. If any
            # of it changes, the previously prepared file is stale -> discard it so
            # a download is never served for the wrong settings. (Speckle removal is
            # automatic inside classes_shapefile_zip, scaled to the class grid, so
            # there's no sieve setting to include here.)
            thr_key = round(threshold, 4) if threshold is not None else None
            shp_sig = (slug, index_name, n_classes, method, thr_key,
                       bool(ground_mask_on), mask_index_name,
                       int(aoi_mask.sum()) if aoi_mask is not None else 0)
            if st.session_state.get("shp_sig") != shp_sig:
                st.session_state.pop("shp_bytes", None)
                st.session_state.pop("shp_name", None)
                st.session_state["shp_sig"] = shp_sig

            if st.button("Prepare class shapefile", use_container_width=True):
                try:
                    with st.spinner("Polygonising classes…"):
                        zbytes = proc.classes_shapefile_zip(
                            class_grid, n_classes, meta,
                            folder_name=f"{slug}_{index_name}_classes")
                    st.session_state["shp_bytes"] = zbytes
                    st.session_state["shp_name"] = f"{slug}_{index_name}_classes.zip"
                    st.session_state["shp_sig"] = shp_sig
                except Exception as e:
                    st.session_state.pop("shp_bytes", None)
                    st.error(f"Shapefile export failed: {e}")

            if st.session_state.get("shp_bytes"):
                st.download_button("⬇ Class shapefile (zip)",
                                   data=st.session_state["shp_bytes"],
                                   file_name=st.session_state.get("shp_name",
                                                                  "classes.zip"),
                                   mime="application/zip", use_container_width=True)
                st.caption("Polygonised, speckle removed, for QGIS.")
            else:
                st.caption("Prepare to generate a download for the current settings.")

    st.divider()
    st.caption(f"Analysis runs on a ~{meta.get('display_shape',[0,0])[0]}px working "
               "copy; the index/class downloads are at that resolution, the ortho "
               "download is full resolution.")


if __name__ == "__main__":
    main()
