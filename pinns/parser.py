import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_yaml(path):
    with open(path) as f:
        params = yaml.load(f, Loader=yaml.Loader)
    return AttrDict(params)
