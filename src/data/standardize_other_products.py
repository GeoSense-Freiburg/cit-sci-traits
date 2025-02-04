from pathlib import Path

import numpy as np

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import create_sample_raster, open_raster, xr_to_raster


def main() -> None:
    cfg = get_config()

    mapping = {"nit": "X14", "nita": "X50", "sla": "X3117"}

    src_dir = Path(
        cfg.raw_dir, "other-trait-maps", "all-prods_stacks_sla-nit-nita_05D_2022-02-14"
    )

    resolutions = ["55km", "111km", "222km"]

    # "all_prods_stacked"
    for res in resolutions:
        ref_r = create_sample_raster(
            resolution=int(res.split("km")[0]) * 1000, crs="EPSG:6933"
        )

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
                    res,
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

    resolutions = ["1km", "22km", "55km", "111km", "222km"]
    r = open_raster(src_path).sel(band=1)

    # The corresponding values should be masked (-2, -1, 100, 0)
    r = r.where(r > 0)
    r = r.rio.write_nodata(np.nan)

    for res in resolutions:
        ref_r = create_sample_raster(
            resolution=int(res.split("km")[0]) * 1000, crs="EPSG:6933"
        )
        r = r.rio.reproject_match(ref_r)
        out_path = Path(
            cfg.interim_dir,
            "other_trait_maps",
            res,
            "X14_moreno.tif",
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
        ref_r = create_sample_raster(
            resolution=int(res.split("km")[0]) * 1000, crs="EPSG:6933"
        )
        r = open_raster(src_path)
        r = r.rio.reproject_match(ref_r)
        out_path = Path(
            cfg.interim_dir,
            "other_trait_maps",
            res,
            "X14_vallicrosa.tif",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Writing {out_path}")
        xr_to_raster(r, out_path)

    src_dir = Path(cfg.raw_dir, "other-trait-maps", "gbif_spo_wolf", "Shrub_Tree_Grass")
    resolutions = ["22km", "55km", "111km", "222km"]
    wolf_resolutions = ["02deg", "05deg", "05deg_to_111km", "2deg"]
    wolf_mapping = {"X14": "X14", "X50": "X50", "X11": "X3117"}
    for res, wolf_res in zip(resolutions, wolf_resolutions):
        ref_r = create_sample_raster(
            resolution=int(res.split("km")[0]) * 1000, crs="EPSG:6933"
        )
        if wolf_res == "05deg_to_111km":
            wolf_res = "05deg"

        for wolf_id, try6_id in wolf_mapping.items():
            r = open_raster(
                src_dir / wolf_res / f"GBIF_TRYgapfilled_{wolf_id}_{wolf_res}.grd"
            ).rio.reproject_match(ref_r)

            out_path = Path(
                cfg.interim_dir, "other_trait_maps", res, f"{try6_id}_wolf.tif"
            )
            log.info(f"Writing {out_path}")
            xr_to_raster(r, out_path)


if __name__ == "__main__":
    main()
