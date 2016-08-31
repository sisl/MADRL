import runners.archs


def get_arch(name):
    constructor = getattr(runners.archs, name)
    return constructor
