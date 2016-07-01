import numpy as np


def rectangle_map(xs, ys, xb=0.3, yb=0.2):
    """
    Returns a 2D 'map' with a rectangle building centered in the middle
    Map is a 2D numpy array
    xb and yb are buffers for each dim representing the raio of the map to leave open on each side
    """
    rmap = np.zeros((xs, ys), dtype=np.int32)
    for i in xrange(xs):
        for j in xrange(ys):
            # are we in the rectnagle in x dim?
            if (float(i)/xs) > xb and (float(i)/xs) < (1.0-xb):
                # are we in the rectangle in y dim?
                if (float(j)/ys) > yb and (float(j)/ys) < (1.0-yb):
                    rmap[i,j] = -1 # -1 is building pixel flag
    return rmap

def cross_map(xs, ys):
    cmap = np.zeros((xs, ys))

     


