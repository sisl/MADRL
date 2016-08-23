import archs


def get_arch(name):
    constructor = getattr(archs, name)
    return constructor
