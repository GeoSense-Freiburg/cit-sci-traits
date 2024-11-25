Combining citizen science and Earth observation to predict global plant trats
==============================


Project Organization
------------

    ├── LICENSE
    ├── README.md                    <- The top-level README for developers using this
    │                                   project.
    ├── data
    │   ├── external                 <- Data from third party sources.
    │   ├── interim                  <- Intermediate data that has been transformed.
    │   ├── processed                <- The final, canonical data sets for modeling.
    │   └── raw                      <- The original, immutable data dump.
    │
    ├── notebooks                    <- Jupyter notebooks. Naming convention is a number
    │                                   (for ordering), the creator's initials, and a
    │                                   short `-` delimited description, e.g.
    │                                   `1-0_jqp_initial-data-exploration`.
    │
    ├── references                   <- Data dictionaries, manuals, and all other
    │                                   explanatory materials.
    │
    ├── reports                      <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                  <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py                     <- makes project pip installable (pip install -e .)
    │                                   so src can be imported
    ├── src                          <- Source code for use in this project.
    │   ├── __init__.py              <- Makes src a Python module
    │   │
    │   ├── data                     <- Scripts to download or generate data
    │   │
    │   ├── features                 <- Scripts to turn raw or interim data into features for modeling
    │   │
    │   └── visualization            <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── tests                        <- Unit tests for use with `pytest`
    │
    ├── pyproject.toml               <- Human-readable project dependencies managed with
    │                                   Poetry
    ├── poetry.lock                  <- File used by Poetry to install dependencies
    ├── conda-linux-64.lock          <- File used by conda-lock to install dependencies for 64-bit Linux systems
    ├── environment.yml              <- File used by conda-lock to specify dependencies
    ├── dvc.yml                      <- DVC pipeline definitions
    ├── params.yml                   <- DVC parameter definitions. **IMPORTANT: this project**
    │                                   **also uses this file as a config file (see src.conf.parse_params)**
    ├── dvc.lock                     <- DVC file which tracks changes to data tracked by DVC
    └── .pre-commit-config.yaml      <- pre-commit Git hooks

    ## Running the project locally
    ### Data access
    #### Predictor data
    The predictor data used for this experiment is all open source. See https://github.com/GeoSense-Freiburg/panops-data-registry for the exact Google Earth Engine scripts used to retrieve the predictor data.

    #### Trait data
    Similarly, the TRY trait data is also available for download via https://trydb.org. See `params.yaml` for a list of traits used.

    #### GBIF species observations
    GBIF observations are also freely available. See https://github.com/GeoSense-Freiburg/panops-data-registry/references/gbif/query_all_tracheophyta.json for the exact query used in this experiment, as well as https://github.com/GeoSense-Freiburg/panops-data-registry/src/gbif/get_gbif_data.py for the download script.

    ## Enabling GPU for model training

    The verison of LightGBM that ships with Autogluon unfortunately doesn't support GPU use. To enable GPU, LightGBM needs to be built and installed locally, per the [documentation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html).

    ```shell
    conda activate <your-environment>
    pip uninstall -y lightgbm
    cd ..
    git clone --recursive https://github.com/microsoft/LightGBM
    cd LightGBM
    cmake -B build -S . -DUSE_GPU=1
    cmake --build build -j$(nproc)
    sh ./build-python.sh install --precompile
    ```
