from __future__ import annotations

import yaml


class AttrDict(dict):
    """Dictionary subclass allowing attribute-style access.

    Enables ``hp.lr`` instead of ``hp['lr']`` for convenience.

    Examples
    --------
    >>> d = AttrDict({'lr': 0.001, 'model': {'name': 'SIREN'}})
    >>> d.lr
    0.001
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_yaml(path: str) -> AttrDict:
    """Load a YAML configuration file as an AttrDict.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    AttrDict
        Configuration dictionary with attribute access.
    """
    with open(path) as f:
        params = yaml.safe_load(f)
    return AttrDict(params)
