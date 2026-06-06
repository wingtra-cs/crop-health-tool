"""
r2.py — storage layer for the PTPN canopy web app.

All access to Cloudflare R2 (S3-compatible object storage) lives here, so the
rest of the app never touches boto3 or the bucket directly. The app calls three
things:

    list_orthos()            -> [{"slug", "name", "meta"}], ready bundles only
    load_bundle(slug)        -> dict of in-memory arrays + metadata (cached)
    presigned_url(key, ttl)  -> temporary direct-download link (bytes go R2->browser)

Design rules baked in here:
  * READ-ONLY credential. This layer never writes to the bucket; uploads are done
    separately from your laptop with a write token. (So there is deliberately no
    put/delete function here.)
  * Readiness filter. A bundle prefix is only "ready" (selectable) once its
    meta.json exists. Half-uploaded bundles are skipped, matching the
    "upload meta.json last" rule.
  * Big bytes never flow through the app. The full ortho is served via a
    presigned URL (browser <-> R2 directly); only the small bundle files are
    pulled into the app's memory.

Secrets are read from st.secrets["r2"] by name — nothing sensitive is in code:

    [r2]
    account_id = "..."
    access_key = "..."   # READ-ONLY token
    secret_key = "..."
    bucket     = "ptpn-bucket"
"""

import io
import json

import numpy as np
import streamlit as st

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


ORTHO_PREFIX = "orthos/"          # top-level prefix that holds one folder per ortho
_BUNDLE_ARRAYS = ("ndvi", "ndre", "cire", "mask_ndvi", "valid")


# --------------------------------------------------------------------------- #
#  Client
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _client():
    """A cached boto3 S3 client pointed at R2. Cached as a resource so a single
    client is shared across reruns and users."""
    s = st.secrets["r2"]
    endpoint = f"https://{s['account_id']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=s["access_key"],
        aws_secret_access_key=s["secret_key"],
        region_name="auto",                      # R2 has no AWS regions; must be "auto"
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def _bucket():
    return st.secrets["r2"]["bucket"]


# --------------------------------------------------------------------------- #
#  Low-level helpers
# --------------------------------------------------------------------------- #
def _get_bytes(key):
    """Fetch an object's raw bytes, or None if it doesn't exist."""
    try:
        obj = _client().get_object(Bucket=_bucket(), Key=key)
        return obj["Body"].read()
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def _key_exists(key):
    try:
        _client().head_object(Bucket=_bucket(), Key=key)
        return True
    except ClientError:
        return False


# --------------------------------------------------------------------------- #
#  Discovery
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def list_orthos():
    """List ready ortho bundles under orthos/.

    Returns a list of {"slug", "name", "meta"} sorted by display name. A prefix is
    included ONLY if it has a readable meta.json (the readiness flag), so a bundle
    that is still uploading never shows up. Cached for 5 minutes; call
    refresh_orthos() to clear immediately after an upload."""
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=_bucket(), Prefix=ORTHO_PREFIX,
                                   Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefix = cp["Prefix"]                      # e.g. "orthos/palm_ortho/"
            slug = prefix[len(ORTHO_PREFIX):].strip("/")
            if not slug:
                continue
            raw = _get_bytes(prefix + "meta.json")
            if raw is None:
                continue                               # not ready yet — skip
            try:
                meta = json.loads(raw.decode("utf-8"))
            except Exception:
                continue                               # malformed — skip defensively
            out.append({"slug": slug,
                        "name": meta.get("name", slug),
                        "meta": meta})
    out.sort(key=lambda d: d["name"].lower())
    return out


def refresh_orthos():
    """Clear the discovery cache so a freshly uploaded (or removed) ortho is
    picked up without waiting for the TTL. Wire this to a 'Refresh' button."""
    list_orthos.clear()


# --------------------------------------------------------------------------- #
#  Bundle loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=True)
def load_bundle(slug):
    """Load one ortho's small artifacts into memory. Cached as a resource keyed by
    slug, so the first user to open an ortho pays the load and the rest reuse it.

    Returns a dict:
        {
          "slug": str,
          "meta": dict,                  # parsed meta.json
          "ndvi","ndre","cire": float32 arrays (display res),
          "mask_ndvi": float32 array,
          "valid": bool array,
          "bounds": dict or None,        # WGS84 bounds for the Leaflet overlay
          "overlay_png": bytes or None,  # web-mercator RGBA overlay
          "transform": tuple,            # affine of the display-res grid
          "ortho_key": str,              # bucket key of the full ortho (for download)
        }
    Raises FileNotFoundError if a required artifact is missing."""
    prefix = f"{ORTHO_PREFIX}{slug}/"

    raw_meta = _get_bytes(prefix + "meta.json")
    if raw_meta is None:
        raise FileNotFoundError(f"{slug}: meta.json not found")
    meta = json.loads(raw_meta.decode("utf-8"))

    bundle = {"slug": slug, "meta": meta,
              "transform": tuple(meta.get("transform", ())),
              "ortho_key": meta.get("ortho_key", prefix + "ortho.tif")}

    # required numpy arrays
    for nm in _BUNDLE_ARRAYS:
        raw = _get_bytes(f"{prefix}{nm}.npy")
        if raw is None:
            raise FileNotFoundError(f"{slug}: {nm}.npy not found")
        bundle[nm] = np.load(io.BytesIO(raw), allow_pickle=False)

    # optional web-overlay assets
    raw_bounds = _get_bytes(prefix + "bounds_4326.json")
    bundle["bounds"] = json.loads(raw_bounds.decode("utf-8")) if raw_bounds else None
    bundle["overlay_png"] = _get_bytes(prefix + "overlay_webmerc.png")  # may be None
    bundle["rgb_png"] = _get_bytes(prefix + "rgb.png")                  # native-grid preview

    return bundle


# --------------------------------------------------------------------------- #
#  Downloads
# --------------------------------------------------------------------------- #
def presigned_url(key, ttl=900):
    """A temporary direct-download URL for an object (default 15 min). The browser
    downloads straight from R2, bypassing this app — so GB-scale orthos never pass
    through the server. Use for the full ortho download button."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": key},
        ExpiresIn=int(ttl),
    )


def ortho_download_url(bundle, ttl=900):
    """Convenience: presigned URL for a loaded bundle's full ortho."""
    return presigned_url(bundle["ortho_key"], ttl=ttl)


# --------------------------------------------------------------------------- #
#  Local smoke test (run directly, NOT under Streamlit):
#      python r2.py
#  Requires a [r2] section in .streamlit/secrets.toml next to this file.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Minimal standalone check that doesn't need the Streamlit runtime.
    import tomllib
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, ".streamlit", "secrets.toml"), "rb") as f:
        secrets = tomllib.load(f)
    s = secrets["r2"]
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{s['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=s["access_key"],
        aws_secret_access_key=s["secret_key"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    r = client.list_objects_v2(Bucket=s["bucket"], Prefix=ORTHO_PREFIX, Delimiter="/")
    prefixes = [p["Prefix"] for p in r.get("CommonPrefixes", [])]
    print("Prefixes under orthos/:", prefixes)
    for pfx in prefixes:
        slug = pfx[len(ORTHO_PREFIX):].strip("/")
        try:
            m = client.get_object(Bucket=s["bucket"], Key=pfx + "meta.json")
            meta = json.loads(m["Body"].read().decode("utf-8"))
            print(f"  {slug}: READY — name={meta.get('name')!r} "
                  f"shape={meta.get('display_shape')}")
        except ClientError:
            print(f"  {slug}: not ready (no meta.json)")
