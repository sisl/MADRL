import h5py
import timeit

class Timer(object):
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



def split_h5_name(fullpath, sep='/'):
    """
    From h5ls.c:
     * Example: ../dir1/foo/bar/baz
     *          \_________/\______/
     *             file       obj
     *
    """
    sep_inds = [i for i, c in enumerate(fullpath) if c == sep]
    for sep_idx in sep_inds:
        filename, objname = fullpath[:sep_idx], fullpath[sep_idx:]
        if not filename: continue
        # Try to open the file. If it fails, try the next separation point.
        try: h5py.File(filename, 'r').close()
        except IOError: continue
        # It worked!
        return filename, objname
    raise IOError('Could not open HDF5 file/object {}'.format(fullpath))
