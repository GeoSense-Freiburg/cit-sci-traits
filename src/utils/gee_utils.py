import json
import subprocess
import tempfile
from pathlib import Path

import google.auth
from google.cloud import storage


def _transfer_gs_asset_to_gee(manifest_fn: str) -> None:
    """Transfer a Google Cloud Storage asset to Google Earth Engine using a manifest file."""
    subprocess.run(
        [
            "earthengine",
            "upload",
            "manifest",
            "--manifest",
            manifest_fn,
        ]
    )


def _build_manifest(asset_id: str, gs_bucket: str, gee_project: str) -> dict:
    """Build a manifest file for a Google Cloud Storage asset.

    Args:
        asset_id: The asset ID in Google Earth Engine.
        gs_bucket: The bucket of the Google Cloud Storage asset.

    Returns:
        A dictionary representing the manifest file.
    """
    return {
        "name": f"{gee_project}/assets",
        "tilesets": [
            {
                "sources": [
                    {
                        "uris": [f"gs://{gs_bucket}/{asset_id}"],
                    }
                ]
            }
        ],
    }


def _write_manifest(manifest: dict) -> str:
    """Write a manifest file to a temporary file and return the file name."""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(manifest, f)
        return f.name


def _create_manifest(asset_id: str, gs_bucket: str, gee_project: str) -> str:
    """Create a manifest file for a Google Cloud Storage asset."""
    manifest = _build_manifest(asset_id, gs_bucket, gee_project)
    manifest_fn = _write_manifest(manifest)
    return manifest_fn


def _delete_manifest(manifest_fn: str) -> None:
    """Delete a manifest file."""
    Path(manifest_fn).unlink()


def _get_asset_ids_from_bucket(bucket_name: str) -> list:
    """Get all asset IDs from a Google Cloud Storage bucket."""
    _, project = google.auth.default()
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    return [blob.name for blob in bucket.list_blobs()]


def transfer_gs_assets_to_gee(gs_bucket: str, gee_project: str) -> None:
    """Transfer all assets from a Google Cloud Storage bucket to Google Earth Engine."""
    asset_idx = _get_asset_ids_from_bucket(gs_bucket)
    for asset_id in asset_idx:
        manifest = _create_manifest(asset_id, gs_bucket, gee_project)
        _transfer_gs_asset_to_gee(manifest)
        _delete_manifest(manifest)
