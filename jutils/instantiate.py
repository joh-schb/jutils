import importlib
from functools import partial


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_partial_from_config(config):
    return partial(get_obj_from_str(config['target']), **config.get('params', dict()))


if __name__ == "__main__":
    # instantiate_from_config(config)
    cfg = dict(target="jutils.helpers.Namespace", params=dict(a=1, b=2))
    print("instantiate_from_config(cfg):", instantiate_from_config(cfg))
