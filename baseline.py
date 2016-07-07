import numpy as np


class Baseline(object):
    def __init__(self, obsfeat_space, action_space):
        self.obsfeat_space = obsfeat_space
        self.action_space = action_space

    def get_params(self):
        raise NotImplementedError()

    def set_params(self, val):
        raise NotImplementedError()

    def fit(self, trajs):
        raise NotImplementedError()

    def predict(self, trajs):
        raise NotImplementedError()


class ZeroBaseline(Baseline):
    def __init__(self, env_spec):
        pass

    def get_params(self):
        return None

    def set_params(self, val):
        pass

    def fit(self, trajs):
        pass

    def predict(self, trajs):
        return np.zeros_like(trajs.r.stacked)


class LinearFeatureBaseline(Baseline):
    def __init__(self, obsfeat_space, action_space, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__(obsfeat_space, action_space)
        self.w_Df = None
        self._reg_coeff = reg_coeff

    def get_params(self):
        return self.w_Df

    def set_params(self, vals):
        self.w_Df = vals

    def _features(self, traj):
        o = np.clip(traj.obsfeat_T_Df, -10, 10)
        l = len(traj)
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al , al**2, al**3, np.ones((l,1))], axis=1)

    def fit(self, trajs, qvals):
        feat_B_Df = np.concatenate([self._features(traj) for traj in trajs])
        self.w_Df = np.linalg.lstsq(
            feat_B_Df.T.dot(feat_B_Df) + self._reg_coeff * np.identity(feat_B_Df.shape[1]),
            feat_B_Df.T.dot(qvals)
        )[0]
        return []

    def predict(self, trajs):
        feat_B_Df = np.concatenate([self._features(traj) for traj in trajs])
        if self.w_Df is None:
            return np.zeros_like(trajs.r.stacked)
        return feat_B_Df.dot(self.w_Df)
