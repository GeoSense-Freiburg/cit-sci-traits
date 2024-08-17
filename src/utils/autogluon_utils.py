"""Autogluon utility functions."""

from pathlib import Path


# TODO: #12 Update get_best_model_ag to use the new model directory structure @dluks
def get_best_model_ag(models_dir: Path) -> Path:
    """Find the best model in the specified directory."""
    quality_levels = ["best", "high", "medium", "good", "fastest"]
    # Initialize the variables to store the best model and its timestamp
    best_model = None

    # Loop over the quality levels in descending order
    for quality in quality_levels:
        # Get the directories for the current quality level
        models = sorted(
            [
                d
                for d in models_dir.iterdir()
                if d.is_dir() and d.name.startswith(quality)
            ],
            reverse=True,
        )

        if not models:
            continue

        best_model = models[0]
        break

    if best_model is None:
        raise ValueError(f"No models found in the specified directory: {models_dir}")

    return best_model
