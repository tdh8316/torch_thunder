import json
from typing import Optional

from torch_thunder.module import ThunderModule


def load_hparams(filename: str) -> dict:
    """
    Load hyperparameters from a JSON file

    Args:
        filename (str): JSON file to load hyperparameters from

    Returns:
        dict: Hyperparameters
    """
    with open(filename, "r") as f:
        hparams = json.load(f)
    return hparams


def save_hparams_from_kwargs(
    filename: str,
    **kwargs,
):
    """
    Save flat hyperparameters to a JSON file

    Args:
        filename (str): JSON file to save hyperparameters to
        **kwargs: Hyperparameters to save
    """
    with open(filename, "w") as f:
        json.dump(kwargs, f, indent=4)


def save_hparams_from_dict(
    filename: str,
    hparams: dict,
):
    """
    Save hyperparameters to a JSON file

    Args:
        filename (str): JSON file to save hyperparameters to
        hparams (dict): Hyperparameters to save
    """
    with open(filename, "w") as f:
        json.dump(hparams, f, indent=4)


def save_hparams_from_model(
    filename: str,
    model: ThunderModule,
):
    """
    Save hyperparameters from a model to a JSON file

    Args:
        filename (str): JSON file to save hyperparameters to
        model (ThunderModule): Model to save hyperparameters from
    """
    _hparams: Optional[dict] = model.hyperparameters()
    if _hparams is None:
        message = (
            "Trying to save hyperparameters from a model, "
            + "but `model.hyperparameters()` returned None. "
            + "To save hyperparameters from a model, "
            + "the model must have a `hyperparameters()` method that returns a dictionary."
        )
        raise ValueError(message)

    save_hparams_from_dict(filename, _hparams)
