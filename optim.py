import numpy as np
import scipy.sparse.linalg as ssl

import nn
import util
import tfutil


def btlinesearch(f, x0, fx0, g, dx, accept_ratio, shrink_factor, max_steps, verbose=False):
    """
    Find a step size t such that f(x0 + t*dx) is within a factor
    accept_ratio of the linearized function value improvement.

    Args:
        f: the function
        x0: starting point for search
        fx0: the value f(x0). Will be computed if set to None.
        g: search direction, typically the gradient of f at x0
        dx: the largest possible step to take
        accept_ratio: termination criterion
        shrink_factor: how much to decrease the step every iteration
        max_steps: max number of tries
    """
    if fx0 is None: fx0 = f(x0)
    t = 1.
    m = g.dot(dx)
    if accept_ratio != 0 and m > 0: print('WARNING: %.10f not <= 0' % m)
    num_steps = 0
    while num_steps < max_steps:
        true_imp = f(x0 + t*dx) - fx0
        lin_imp = t*m
        if verbose: print(true_imp, lin_imp, accept_ratio)
        if true_imp <= accept_ratio * lin_imp:
            break
        t *= shrink_factor
        num_steps += 1
    return x0 + t*dx, num_steps


def numdiff_hvp(v, grad_func, x0, grad0=None, finitediff_delta=1e-4):
    """
    Approximate Hessian-vector product.

    Uses a 1-dimensional finite difference approximation for the
    directional derivative of the gradient function.
    if grad0 given:
         (g(x+rv)-g(x))/r
    Otherwise
         (g(x+rv)-g(x-rv))/2r

    Args:
        v: the vector to left-multiply by the Hessian
        grad_func: gradient function
        x0: point at which to evaluate the Hessian
        grad0: should equal grad_func(x0), or None to compute in here.
        finitediff_delta: step size for finite difference
    """
    assert v.shape == x0.shape
    if np.allclose(v, 0): return np.zeros_like(v)
    eps = finitediff_delta / np.linalg.norm(v)
    dx = eps * v
    grad1 = grad_func(x0+dx)
    if grad0 is None:
        grad0 = grad_func(x0-dx)
        eps = 2*eps
    out = grad1 - grad0; out /= eps
    return out


def ngstep(x0, obj0, objgrad0, obj_and_kl_func, hvpx0_func, max_kl, damping, max_cg_iter, enable_bt):
    '''
    Natural gradient step using hessian-vector products

    Args:
        x0: current point
        obj0: objective value at x0
        objgrad0: grad of objective value at x0
        obj_and_kl_func: function mapping a point x to the objective and kl values
        hvpx0_func: function mapping a vector v to the KL Hessian-vector product H(x0)v
        max_kl: max kl divergence limit. Triggers a line search.
        damping: multiple of I to mix with Hessians for Hessian-vector products
        max_cg_iter: max conjugate gradient iterations for solving for natural gradient step
    '''

    assert x0.ndim == 1 and x0.shape == objgrad0.shape

    # Solve for step direction
    damped_hvp_func = lambda v: hvpx0_func(v) + damping*v
    hvpop = ssl.LinearOperator(shape=(x0.shape[0], x0.shape[0]), matvec=damped_hvp_func)
    step, _ = ssl.cg(hvpop, -objgrad0, maxiter=max_cg_iter)
    fullstep = step / np.sqrt(.5 * step.dot(damped_hvp_func(step)) / max_kl + 1e-8)

    # Line search on objective with a hard KL wall
    if not enable_bt:
        return x0+fullstep, 0

    def barrierobj(p):
        obj, kl = obj_and_kl_func(p)
        return np.inf if kl > 2*max_kl else obj

    xnew, num_bt_steps = btlinesearch(
        f=barrierobj,
        x0=x0,
        fx0=obj0,
        g=objgrad0,
        dx=fullstep,
        accept_ratio=.1, shrink_factor=.5, max_steps=10)
    return xnew, num_bt_steps


def make_ngstep_func(model, compute_obj_kl, compute_obj_kl_with_grad, compute_hvp_helper):
    """Make a wrapper for ngstep for classes that implement nn.Model

    Subsamples inputs for faster Hessian-vector products
    """
    assert isinstance(model, nn.Model)

    def wrapper(sess, feed, max_kl, damping,
                subsample_hvp_frac=.1, grad_stop_tol=1e-6, max_cg_iter=10, enable_bt=True):
        assert isinstance(feed, tuple)

        params0 = model.get_params(sess)
        obj0, kl0, objgrad0 = compute_obj_kl_with_grad(sess, *feed)
        gnorm = util.maxnorm(objgrad0)
        assert np.allclose(kl0, 0., atol=1e-6), 'Initial KL divergence is %.7f, but should be 0' % (kl0)

        # Terminate early when gradient too small
        if gnorm < grad_stop_tol:
            obj1, kl1 = obj0, kl0
            num_bt_steps = 0
        else:
            # Data subsampling for hvp
            subsamp_feed = feed if subsample_hvp_frac is None else tfutil.subsample_feed(feed, subsample_hvp_frac)

            def hvpx0_func(v):
                def klgrad_func(p):
                    with model.try_params(sess, params0):
                        klgrad = compute_hvp_helper(sess, *subsamp_feed)
                        return klgrad
                return numdiff_hvp(v, klgrad_func, params0)

            # Line search objective
            def obj_and_kl_func(p):
                with model.try_params(sess, p):
                    obj, kl = compute_obj_kl(sess, *feed)
                return -obj, kl

            params1, num_bt_steps = ngstep(
                x0=params0,
                obj0=-obj0,
                objgrad0=-objgrad0,
                obj_and_kl_func=obj_and_kl_func,
                hvpx0_func=hvpx0_func,
                max_kl=max_kl,
                damping=damping,
                max_cg_iter=max_cg_iter,
                enable_bt=enable_bt
            )
            model.set_params(sess, params1)
            obj1, kl1 = compute_obj_kl(sess, *feed)
        return [
            ('dl', obj1 - obj0, float), # improvement of objective
            ('kl', kl1, float),         # kl cost of solution
            ('gnorm', gnorm, float),
            ('bt', num_bt_steps, int),  # number of backtracking steps
        ]

    return wrapper
