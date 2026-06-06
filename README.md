# Crop Health Visualization and Classification Tool

A hosted web app for exploring multispectral orthomosaics (5-band MicaSense
RedEdge-P: B-G-R-RedEdge-NIR, calibrated reflectance). Users pick a pre-uploaded
orthomosaic, tune a bare-earth threshold, view a vegetation index, classify the
canopy into relative-vigour bands, and download the data. It is a scouting and
visualization aid, not a diagnosis.

## How it fits together

Three tiers:

1. **Cloud Object Storage (Cloudflare R2)** — holds the orthos and their
   preprocessed "bundles". The source of truth for which datasets exist.
2. **This Streamlit app (Community Cloud)** — holds no data. It lists the bucket,
   loads small precomputed arrays for the selected ortho, runs the interactive
   threshold / index / classification on those, and serves downloads.
3. **The browser** — renders the UI and downloads the full ortho *directly* from
   storage via a short-lived presigned URL (large files never pass through the app).

Heavy, one-time work (reading the multi-GB ortho, computing indices, building the
web overlay) happens offline in `preprocess_ortho.py`, NOT in this app.

## Repo layout

    streamlit_app.py        # entry point (Community Cloud runs this)
    r2.py                   # storage layer: list_orthos / load_bundle / presigned_url
    processing.py           # index / threshold / classification / exports
    requirements.txt        # pip dependencies
    packages.txt            # apt deps (GDAL safety net for rasterio/geopandas)
    .gitignore
    .streamlit/
        config.toml         # theme (committed)
        secrets.toml        # LOCAL ONLY, gitignored — see secrets.toml.example
    README.md

Not in the repo: the orthos/bundles (they live in R2) and `preprocess_ortho.py`
(a local tool you run on your machine to build bundles).

## Secrets

Local runs read `.streamlit/secrets.toml` (copy from `secrets.toml.example`).
On Community Cloud, paste the same contents into Settings -> Secrets.
The R2 credentials must be a READ-ONLY token. Required keys:

    app_password
    [r2] account_id, access_key, secret_key, bucket

## Local development

    python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    # create .streamlit/secrets.toml from the example, with real read-only keys
    python r2.py            # smoke test: lists ready bundles in the bucket
    streamlit run streamlit_app.py

## Adding / removing a dataset (no code change, no redeploy)

Build a bundle locally:

    python preprocess_ortho.py /path/to/ortho.tif --name "Field 3 — North"

Upload it to the bucket with the readiness rule:

    # everything EXCEPT meta.json first
    rclone copy ./bundles/field_3_north r2:ptpn-bucket/orthos/field_3_north \
        --exclude meta.json --progress
    # meta.json LAST — this is what makes the bundle appear in the app
    rclone copy ./bundles/field_3_north/meta.json \
        r2:ptpn-bucket/orthos/field_3_north/ --progress

To remove a dataset: delete `meta.json` FIRST, then the rest of the folder.
Uploads use a separate WRITE token kept only on your machine — never the app's
read-only token, and never committed.

## Deploy

Push to GitHub, then on Streamlit Community Cloud: New app -> point at the repo,
branch, and `streamlit_app.py` -> paste secrets -> deploy. Each push redeploys.
