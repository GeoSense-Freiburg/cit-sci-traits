from pathlib import Path

import numpy as np

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import create_sample_raster, open_raster, xr_to_raster


def main() -> None:
    cfg = get_config()

    mapping = {"nit": "X14", "nita": "X50", "sla": "X11"}

    src_dir = Path(
        cfg.raw_dir, "other-trait-maps", "all-prods_stacks_sla-nit-nita_05D_2022-02-14"
    )

    resolutions = [0.5, 1, 2]

    # "all_prods_stacked"
    for res in resolutions:
        ref_r = create_sample_raster(resolution=res, crs="EPSG:4326")

        for trait, code in mapping.items():
            all_prods_nitm = open_raster(
                Path(src_dir, f"all-prods_{trait}_stack_all-maps_05D_2022-02-14.grd")
            )
            authors = all_prods_nitm.attrs["long_name"]
            names = [n.lower() for n in authors]

            for i, _ in enumerate(all_prods_nitm):
                if names[i] in ("moreno", "vallicrosa"):
                    continue  # We received a separate file for these authors

                out_path = Path(
                    cfg.interim_dir,
                    "other_trait_maps",
                    str(res).replace(".", ""),
                    f"{code}_{names[i]}.tif",
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)

                log.info(f"Writing {out_path}")
                r = all_prods_nitm.sel(band=i + 1)
                r = r.rio.reproject_match(ref_r)
                r.attrs["long_name"] = names[i]
                xr_to_raster(r, out_path)

    # Moreno
    src_path = Path(
        cfg.raw_dir, "other-trait-maps", "AMM_Trait_maps_v3_2023", "LNC_1km_v3.tif"
    )

    resolutions = [0.01, 0.2, 0.5, 1, 2]
    r = open_raster(src_path).sel(band=1)

    # The corresponding values should be masked (-2, -1, 100, 0)
    r = r.where(r > 0)
    r = r.rio.write_nodata(np.nan)

    for res in resolutions:
        ref_r = create_sample_raster(resolution=res, crs="EPSG:4326")
        r = r.rio.reproject_match(ref_r)
        out_path = Path(
            cfg.interim_dir,
            "other_trait_maps",
            str(res).replace(".", ""),
            f"X14_moreno.tif",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing {out_path}")
        xr_to_raster(r, out_path)

    # Vallicrosa
    src_path = Path(
        cfg.raw_dir,
        "other-trait-maps",
        "vallicrosa_n_mean",
        "N_mean_predictmap_NewPhy2.grd",
    )

    for res in resolutions:
        ref_r = create_sample_raster(resolution=res, crs="EPSG:4326")
        r = open_raster(src_path)
        r = r.rio.reproject_match(ref_r)
        out_path = Path(
            cfg.interim_dir,
            "other_trait_maps",
            str(res).replace(".", ""),
            f"X14_vallicrosa.tif",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing {out_path}")
        xr_to_raster(r, out_path)

    # TODO: Include Wolf maps?
    # src_dir = Path(cfg.raw_dir, "other-trait-maps", "gbif_spo_wolf", "Shrub_Tree_Grass")
    # resolutions = [0.2, 0.5, 2]


if __name__ == "__main__":
    main()
