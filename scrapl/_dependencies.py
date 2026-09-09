from importlib import import_module


def require_backend(name):
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise ModuleNotFoundError(
            f"The {name} backend requires the optional '{name}' extra. "
            f"Install it with pip install 'scrapl-loss[{name}]', "
            f"or pip install -e '.[{name}]' from a source checkout.",
            name=name,
        ) from error
