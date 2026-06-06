"""
streamlit_app.py — Crop Health Visualization and Classification Tool

Hosted web front-end (Streamlit Community Cloud). This is the APP SHELL only:
  * shared-password gate
  * ortho dropdown driven live by what's in the R2 bucket (via r2.list_orthos)
  * bundle load for the selected ortho (via r2.load_bundle)
  * a quick "it works" panel: metrics, the RGB preview, and a presigned
    download link for the full ortho

The interactive analysis (threshold / index / classification), the map overlay,
and the index / shapefile downloads are added in later steps. Storage access is
entirely in r2.py; processing logic will live in processing.py.

Secrets (Community Cloud "Secrets" box, or local .streamlit/secrets.toml):

    app_password = "..."

    [r2]
    account_id = "..."
    access_key = "..."   # READ-ONLY token
    secret_key = "..."
    bucket     = "ptpn-bucket"
"""

import io
import hmac

import streamlit as st

import r2


APP_TITLE = "Crop Health Visualization and Classification Tool"


# --------------------------------------------------------------------------- #
#  Password gate
#  A soft gate: deters casual access. The real protection on the data is that
#  the bucket is private and downloads use short-lived presigned URLs.
# --------------------------------------------------------------------------- #
def check_password():
    if st.session_state.get("auth_ok"):
        return True

    st.title(APP_TITLE)
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
#  Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_password():
        st.stop()

    st.title(APP_TITLE)

    # ---- Ortho selection (driven live by the bucket) ----------------- #
    with st.sidebar:
        st.header("Dataset")
        cols = st.columns([3, 1])
        with cols[1]:
            if st.button("↻", help="Refresh the dataset list from storage"):
                r2.refresh_orthos()
                st.rerun()

    try:
        orthos = r2.list_orthos()
    except Exception as e:
        st.error("Could not reach storage. Check the app's R2 credentials.")
        st.caption(f"Details: {e}")
        st.stop()

    if not orthos:
        st.info("No datasets are available yet. Once an orthomosaic bundle is "
                "uploaded to storage, it will appear here automatically.")
        st.stop()

    names = [o["name"] for o in orthos]
    with st.sidebar:
        picked = st.selectbox("Orthomosaic", names, index=0)
    selected = orthos[names.index(picked)]
    slug = selected["slug"]

    # ---- Load the selected bundle ------------------------------------ #
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

    # ---- Shell "it works" panel (placeholder for the real analysis) -- #
    st.subheader(meta.get("name", slug))

    c1, c2, c3, c4 = st.columns(4)
    fs = meta.get("full_shape", [None, None])
    ds = meta.get("display_shape", [None, None])
    c1.metric("Full size", f"{fs[1]} × {fs[0]} px" if fs[0] else "—")
    c2.metric("Working size", f"{ds[1]} × {ds[0]} px" if ds[0] else "—")
    c3.metric("Decimation", f"1/{meta.get('decimation', '—')}")
    c4.metric("CRS", str(meta.get("crs") or "—"))

    # RGB preview if the bundle carries one (overlay PNG is web-mercator; the
    # native rgb.png isn't in the bundle dict, so show the overlay if present).
    if bundle.get("overlay_png"):
        st.image(io.BytesIO(bundle["overlay_png"]),
                 caption="True-colour preview (web-mercator overlay)",
                 use_container_width=True)
    else:
        st.caption("No preview overlay in this bundle "
                   "(source may have had no CRS).")

    # Full-ortho download via a short-lived presigned URL (bytes go storage ->
    # browser directly; they never pass through this app).
    try:
        url = r2.ortho_download_url(bundle, ttl=900)
        st.link_button("⬇ Download full 5-band orthomosaic", url)
        st.caption("Direct download from storage; link valid for ~15 minutes.")
    except Exception as e:
        st.caption(f"Ortho download link unavailable: {e}")

    st.divider()
    st.caption("Shell only — interactive threshold, index visualization, "
               "classification, map overlay, and derived-product downloads are "
               "added in the next steps.")


if __name__ == "__main__":
    main()
