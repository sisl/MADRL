from contextlib import contextmanager

import numpy as np
import scipy.linalg
import tensorflow as tf

import optim
import nn
import tfutil
import util


class Baseline(object):
    def __init__(self, obsfeat_space):
        self.obsfeat_space = obsfeat_space

    def get_params(self):
        raise NotImplementedError()

    def set_params(self, val):
        raise NotImplementedError()

    def fit(self, trajs):
        raise NotImplementedError()

    def predict(self, trajs):
        raise NotImplementedError()


class ZeroBaseline(Baseline):
    def __init__(self, obsfeat_space):
        pass

    def get_params(self):
        return None

    def set_params(self, val):
        pass

    def fit(self, trajs):
        return []

    def predict(self, trajs):
        return np.zeros_like(trajs.r.stacked)


class LinearFeatureBaseline(Baseline):
    def __init__(self, obsfeat_space, enable_obsnorm, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__(obsfeat_space)
        self.w_Df = None
        self._reg_coeff = reg_coeff
        with tf.variable_scope('obsnorm'):
            self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.shape[0])

    def get_params(self):
        return self.w_Df

    def set_params(self, vals):
        self.w_Df = vals

    # def _features(self, traj):
    #     o = np.clip(traj.obsfeat_T_Df, -10, 10)
    #     l = len(traj)
    #     al = np.arange(l).reshape(-1, 1) / 100.0
    #     return np.concatenate([o, o**2, al , al**2, al**3, np.ones((l,1))], axis=1)

    # def fit(self, trajs, qvals):
    #     feat_B_Df = np.concatenate([self._features(traj) for traj in trajs])
    #     self.w_Df = np.linalg.lstsq(
    #         feat_B_Df.T.dot(feat_B_Df) + self._reg_coeff * np.identity(feat_B_Df.shape[1]),
    #         feat_B_Df.T.dot(qvals)
    #     )[0]
    #     return []

    # def predict(self, trajs):
    #     feat_B_Df = np.concatenate([self._features(traj) for traj in trajs])
    #     if self.w_Df is None:
    #         return np.zeros_like(trajs.r.stacked)
    #     return feat_B_Df.dot(self.w_Df)
    def update_obsnorm(self, sess, obs_B_Do):
        """Update norms using moving avg"""
        self.obsnorm.update(sess, obs_B_Do)
            
    def _features(self, trajs):
        return np.concatenate([
            trajs.obsfeat.stacked,
            trajs.time.stacked[:,None]/100.,
            (trajs.time.stacked[:,None]/100.)**2,
            np.ones((trajs.obsfeat.stacked.shape[0],1))
        ], axis=1)

    def fit(self, sess, trajs, qvals):
        assert qvals.shape == (trajs.obsfeat.stacked.shape[0],)
        feat_B_Df = self._features(trajs)
        self.w_Df = scipy.linalg.solve(
            feat_B_Df.T.dot(feat_B_Df) + self._reg_coeff*np.eye(feat_B_Df.shape[1]),
            feat_B_Df.T.dot(qvals),
            sym_pos=True
        )
        return []

    def predict(self, sess, trajs):
        feat_B_Df = self._features(trajs)
        if self.w_Df is None:
            self.w_Df = np.zeros(feat_B_Df.shape[1], dtype=trajs.obsfeat.stacked.dtype)
        return feat_B_Df.dot(self.w_Df)


class MLPBaseline(Baseline, nn.Model):
    def __init__(self, obsfeat_space, hidden_spec, enable_obsnorm, enable_vnorm, max_kl, time_scale, varscope_name):
        self.obsfeat_space = obsfeat_space
        self.hidden_spec = hidden_spec
        self.enable_obsnorm = enable_obsnorm
        self.enable_vnorm = enable_vnorm
        self.max_kl = max_kl
        self.time_scale = time_scale

        with tf.variable_scope(varscope_name) as self.varscope:
            with tf.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else nn.NoOpStandardizer)(self.obsfeat_space.shape[0])
            with tf.variable_scope('vnorm'):
                self.vnorm = (nn.Standardizer if enable_vnorm else nn.NoOpStandardizer)(1)

            batch_size = None
            self.obsfeat_B_Df = tf.placeholder(tf.float32, [batch_size, self.obsfeat_space.shape[0]], name='obsfeat_B_Df') # FIXME shape
            self.t_B_1 = tf.placeholder(tf.float32, [batch_size,1], name='t_B')
            scaled_t_B_1 = self.t_B_1 * self.time_scale
            net_input = tf.concat(1, [self.obsfeat_B_Df, scaled_t_B_1])
            with tf.variable_scope('hidden'):
                net = nn.FeedforwardNet(net_input, (self.obsfeat_space.shape[0]+1,), self.hidden_spec)
            with tf.variable_scope('out'):
                out_layer = nn.AffineLayer(net.output, net.output_shape, (1,), initializer=tf.zeros_initializer)
                assert out_layer.output_shape == (1,)
            self.val_B = out_layer.output[:,0]
        # Only code above has trainable vars
        self.param_vars = self.get_trainable_variables()
        self._num_params = self.get_num_params()
        self._curr_params_P = tfutil.flatcat(self.param_vars)

        # Squared loss for fitting the value function
        self.target_val_B = tf.placeholder(tf.float32, [batch_size], name='target_val_B')
        self.obj = -tf.reduce_mean(tf.square(self.val_B-self.target_val_B))
        self.objgrad_P = tfutil.flatcat(tf.gradients(self.obj, self.param_vars))

        # KL divergence (as Gaussian) and its gradient
        self.old_val_B = tf.placeholder(tf.float32, [batch_size], name='old_val_B')
        self.kl = tf.reduce_mean(tf.square(self.old_val_B-self.val_B))

        # KL gradient
        self.kl_grad_P = tfutil.flatcat(tf.gradients(self.kl, self.param_vars))

        # Writing params
        self._flatparams_P = tf.placeholder(tf.float32, [self._num_params], name='flatparams_P')
        self._assign_params = tfutil.unflatten_into_vars(self._flatparams_P, self.param_vars)

    def compute_obj_kl(self, sess, obsfeat_B_Df, t_B, target_val_B, old_val_B):
        return sess.run([self.obj, self.kl], { self.obsfeat_B_Df: obsfeat_B_Df,
                                               self.t_B_1: t_B[:, None],
                                               self.target_val_B: target_val_B,
                                               self.old_val_B: old_val_B })

    def compute_obj_kl_with_grad(self, sess, obsfeat_B_Df, t_B, target_val_B, old_val_B):
        return sess.run([self.obj, self.kl, self.objgrad_P], { self.obsfeat_B_Df: obsfeat_B_Df,
                                                               self.t_B_1: t_B[:, None],
                                                               self.target_val_B: target_val_B,
                                                               self.old_val_B: old_val_B })

    def compute_klgrad(self, sess, obsfeat_B_Df, t_B, target_val_B, old_val_B):
        return sess.run([self.kl_grad_P], { self.obsfeat_B_Df: obsfeat_B_Df,
                                            self.t_B_1: t_B[:, None],
                                            self.target_val_B: target_val_B,
                                            self.old_val_B: old_val_B })[0]

    def _step(self, sess, obs_B_Do, t_B, target_val_B, old_val_B,
              subsample_hvp_frac=.25, damping=1e-2, grad_stop_tol=1e-6, max_cg_iter=10, enable_bt=True):

        params0 = self.get_params(sess)
        feed = (obs_B_Do, t_B, target_val_B, old_val_B)
        obj0, kl0, objgrad0 = self.compute_obj_kl_with_grad(sess, *feed)
        gnorm = util.maxnorm(objgrad0)
        assert np.allclose(kl0, 0.0, atol=1e-05), "Initial KL divergence is %.7f, but should be 0" % (kl0)
        # Terminate early
        if gnorm < grad_stop_tol:
            obj1 = obj0
            kl1 = kl0
            num_bt_steps = 0
        else:
            # Data subsample for Hessain vector products
            subsamp_feed = feed if subsample_hvp_frac is None else tfutil.subsample_feed(feed, subsample_hvp_frac)
            def hvp_klgrad_func(p):
                with self.try_params(sess, p):
                    return self.compute_klgrad(sess, *subsamp_feed)
            # Line search objective
            def obj_and_kl_func(p):
                with self.try_params(sess, p):
                    obj, kl = self.compute_obj_kl(sess, *feed)
                return -obj, kl

            params1, num_bt_steps = optim.ngstep(
                x0=params0,
                obj0=-obj0,
                objgrad0=-objgrad0,
                obj_and_kl_func=obj_and_kl_func,
                hvpx0_func=hvp_klgrad_func,
                max_kl=self.max_kl,
                damping=damping,
                max_cg_iter=max_cg_iter,
                enable_bt=enable_bt)
            self.set_params(sess, params1)
            obj1, kl1 = self.compute_obj_kl(sess, *feed)

        return [
            ('vf_dl', obj1-obj0, float),
            ('vf_kl', kl1, float),
            ('vf_gnorm', gnorm, float),
            ('vf_bt', num_bt_steps, int)
        ]

    def update_obsnorm(self, sess, obs_B_Do):
        """Update norms using moving avg"""
        self.obsnorm.update(sess, obs_B_Do)

    def get_params(self, sess):
        params_P = sess.run(self._curr_params_P)
        assert params_P.shape == (self._num_params,)
        return params_P

    def set_params(self, sess, params_P):
        sess.run(self._assign_params, {self._flatparams_P: params_P})

    @contextmanager
    def try_params(self, sess, params_P):
        orig_params_P = self.get_params(sess)
        self.set_params(sess, params_P)
        yield
        self.set_params(sess, orig_params_P)

    def _predict_raw(self, sess, obsfeat_B_Df, t_B):
        return sess.run(self.val_B, {self.obsfeat_B_Df: obsfeat_B_Df, self.t_B_1: t_B[:, None]})

    def fit(self, sess, trajs, qvals):
        obs_B_Do = trajs.obs.stacked
        t_B = trajs.time.stacked

        # Update norm
        self.obsnorm.update(sess, obs_B_Do)
        self.vnorm.update(sess, qvals[:, None])

        # Take step
        sobs_B_Do = self.obsnorm.standardize(sess, obs_B_Do)
        sqvals_B = self.vnorm.standardize(sess, qvals[:, None])[:, 0]
        feed = (sobs_B_Do, t_B, sqvals_B, self._predict_raw(sess, sobs_B_Do, t_B))
        stepinfo = self._step(sess, *feed)
        return stepinfo

    def predict(self, sess, trajs):
        obs_B_Do = trajs.obs.stacked
        t_B = trajs.time.stacked
        sobs_B_Do = self.obsnorm.standardize(sess, obs_B_Do)
        pred_B = self.vnorm.unstandardize(sess, self._predict_raw(sess, sobs_B_Do, t_B)[:, None])[:,0]
        assert pred_B.shape == t_B.shape
        return pred_B
