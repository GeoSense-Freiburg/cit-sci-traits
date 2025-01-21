import json

from src.utils.dataset_utils import get_final_fns, get_processed_dir
from src.utils.raster_utils import open_raster
from src.utils.trait_utils import get_trait_number_from_id, load_trait_mapping


def build_metadata() -> None:
    """
    Build metadata for the final dataset. This metadata will be used for viewing in the
    GEE web app. The metadata will be saved in the processed/final directory as metadata.json.

    Metadata structure:
    {
        "PFT": {
            <pft>: {
                <trait_num>: {
                    value: {
                        name: <short_name>,
                        long_name: <long_name>,
                        try_id: <trait_num>,
                        unit: <unit>,
                        scale: <scale>,
                        offset: <offset>,
                        min: <min>,  # 2nd percentile
                        max: <max>   # 98th percentile
                    },
                    cov: {
                        name: Coefficient of variation,
                        scale: <scale>,
                        offset: <offset>,
                        min: <min>, # minimum
                        max: <max>  # 98th percentile
                    }
                },
                <trait_num>: {...}
            <pft>: {...}
    }
    """

    fns = list(get_final_fns())
    mapping = load_trait_mapping()

    meta = {"PFT": {}}
    for fn in fns:
        # Get or create the PFT
        pft = fn.parents[2].stem
        meta_pfts = meta["PFT"]
        if pft not in meta_pfts:
            meta["PFT"][pft] = {}
        pft_dict = meta["PFT"][pft]

        # Get or create the trait
        trait_num = get_trait_number_from_id(fn.stem)
        if trait_num not in pft_dict:
            meta["PFT"][pft][trait_num] = {}
        trait_dict = meta["PFT"][pft][trait_num]

        # Fill in the trait and cov values
        r = open_raster(fn)
        value = r.sel(band=1)
        cov = r.sel(band=2)
        trait_dict["value"] = {
            "name": mapping[trait_num]["short"],
            "long_name": mapping[trait_num]["long"],
            "try_id": trait_num,
            "unit": mapping[trait_num]["unit"],
            "scale": r.scales[0],
            "offset": r.offsets[0],
            "min": value.quantile(0.02).values.tolist(),
            "max": value.quantile(0.98).values.tolist(),
        }
        trait_dict["cov"] = {
            "name": "Coefficient of variation",
            "scale": r.scales[1],
            "offset": r.offsets[1],
            "min": cov.min().values.tolist(),
            "max": cov.quantile(0.98).values.tolist(),
        }

    out_fn = get_processed_dir() / "final" / "metadata.json"
    with open(out_fn, "w") as f:
        json.dump(meta, f, indent=4)


if __name__ == "__main__":
    build_metadata()
