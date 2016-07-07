from contextlib import contextmanager
import h5py
import hashlib
import json
import numpy as np
import os
import os.path
import tensorflow as tf


class Model(object):
    def get_variables(self):
        """Get all variables in the graph"""
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.varscope.name)

    def get_trainable_variables(self):
        """Get trainable variables in the graph"""
        assert self.varscope
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.varscope.name)

    def get_num_params(self):
        return sum(v.get_shape().num_elements() for v in self.get_trainable_variables())

    @staticmethod
    def _hash_name2array(name2array):
        def hash_array(a):
            return '%.10f,%.10f,%d' % (np.mean(a), np.var(a), np.argmax(a))
        return hashlib.sha1('|'.join('%s %s' for n, h in sorted([(name, hash_array(a)) for name, a in name2array]))).hexdigest()

    def savehash(self, sess):
        '''Hash is based on values of TRAINABLE variables only'''
        vars_ = self.get_trainable_variables()
        vals = sess.run(vars_)
        return self._hash_name2array([(v.name, val) for v, val in util.safezip(vars_, vals)])

    # HDF5 saving and loading
    # The hierarchy in the HDF5 file reflects the hierarchy in the Tensorflow graph.
    def save_h5(self, sess, h5file, key, extra_attrs=None):
        with h5py.File(h5file, 'a') as f:
            if key in f:
                print('WARNING: key {} already exists in {}'.format(key, h5file))
                dset = f[key]
            else:
                dset = f.create_group(key)

            vs = self.get_trainable_variables()
            vals = sess.run(vs)

            for v, val in util.safezip(vs, vals):
                dset[v.name] = val

            dset.attrs['hash'] = self.savehash(sess)
            if extra_attrs is not None:
                for k, v in extra_attrs:
                    if k in dset.attrs:
                        print('Warning: attribute {} already exists in {}'.format(k, dset.name))
                        dset.attrs[k] = v

    def load_h5(self, sess, h5file, key):
        with h5py.File(h5file, 'r') as f:
            dset = f[key]

            ops = []
            for v in self.get_trainable_variables():
                print('Reading {}'.format(v.name))
                ops.append(v.assign(dset[v.name][...]))
                sess.run(ops)

            h = self.savehash(sess)
            assert h == dset.attrs['hash'], 'Checkpoint hash %s does not match loaded hash %s' % (dset.attrs['hash'], h)

# Layers for feedforward networks

class Layer(Model):
    @property
    def output(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        """Shape refers to the shape without the batch axis, which always implicitly goes first"""
        raise NotImplementedError


class ReshapeLayer(Layer):
    def __init__(self, input_, new_shape):
        self._output_shape = tuple(new_shape)
        print('Reshape(new_shape=%s)' % (str(self._output_shape),))
        with tf.variable_scope(type(self).__name__) as self.varscope:
            self._output = tf.reshape(input_, (-1,)+self._output_shape)
    @property
    def output(self): return self._output
    @property
    def output_shape(self): return self._output_shape


class AffineLayer(Layer):
    def __init__(self, input_B_Di, input_shape, output_shape, initializer):
        assert len(input_shape) == len(output_shape) == 1
        print('Affine(in=%d, out=%d)' % (input_shape[0], output_shape[0]))
        self._output_shape = (output_shape[0],)
        with tf.variable_scope(type(self).__name__) as self.varscope:
            if initializer is None:
                initializer = tf.truncated_normal_initializer(mean=0., stddev=np.sqrt(2./input_shape[0]))
                self.W_Di_Do = tf.get_variable('W', shape=[input_shape[0], output_shape[0]], initializer=initializer)
                self.b_1_Do = tf.get_variable('b', shape=[1, output_shape[0]], initializer=tf.constant_initializer(0.))
                self.output_B_Do = tf.matmul(input_B_Di, self.W_Di_Do) + self.b_1_Do
    @property
    def output(self): return self.output_B_Do
    @property
    def output_shape(self): return self._output_shape


class NonlinearityLayer(Layer):
    def __init__(self, input_B_Di, output_shape, func):
        print('Nonlinearity(func=%s)' % func)
        self._output_shape = output_shape
        with tf.variable_scope(type(self).__name__) as self.varscope:
            self.output_B_Do = {'relu': tf.nn.relu, 'elu': tf.nn.elu, 'tanh': tf.tanh}[func](input_B_Di)
    @property
    def output(self): return self.output_B_Do
    @property
    def output_shape(self): return self._output_shape


class ConvLayer(Layer):
    def __init__(self, input_B_Ih_Iw_Ci, input_shape, Co, Fh, Fw, Oh, Ow, Sh, Sw, padding, initializer):
        # TODO: calculate Oh and Ow from the other stuff.
        assert len(input_shape) == 3
        Ci = input_shape[2]
        print('Conv(chanin=%d, chanout=%d, filth=%d, filtw=%d, outh=%d, outw=%d, strideh=%d, stridew=%d, padding=%s)' % (Ci, Co, Fh, Fw, Oh, Ow, Sh, Sw, padding))
        self._output_shape = (Oh, Ow, Co)
        with tf.variable_scope(type(self).__name__) as self.varscope:
            if initializer is None:
                initializer = tf.truncated_normal_initializer(mean=0., stddev=np.sqrt(2./(Fh*Fw*Ci)))
                self.W_Fh_Fw_Ci_Co = tf.get_variable('W', shape=[Fh, Fw, Ci, Co], initializer=initializer)
                self.b_1_1_1_Co = tf.get_variable('b', shape=[1, 1, 1, Co], initializer=tf.constant_initializer(0.))
                self.output_B_Oh_Ow_Co = tf.nn.conv2d(input_B_Ih_Iw_Ci, self.W_Fh_Fw_Ci_Co, [1, Sh, Sw, 1], padding) + self.b_1_1_1_Co
    @property
    def output(self): return self.output_B_Oh_Ow_Co
    @property
    def output_shape(self): return self._output_shape

def _check_keys(d, keys, optional):
    s = set(d.keys())
    if not (s == set(keys) or s == set(keys+optional)):
        raise RuntimeError('Got keys %s, but expected keys %s with optional keys %s' % (str(s, str(keys), str(optional))))


def _parse_initializer(layerspec):
    if 'initializer' not in layerspec:
        return None
    initspec = layerspec['initializer']
    raise NotImplementedError('Unknown layer initializer type %s' % initspec['type'])


class FeedforwardNet(Layer):
    def __init__(self, input_B_Di, input_shape, layerspec_json):
        """        
        Args:
            layerspec (string): JSON string describing layers
        """        
        assert len(input_shape) >= 1
        self.input_B_Di = input_B_Di

        layerspec = json.loads(layerspec_json)
        print('Loading feedforward net specification')
        print(json.dumps(layerspec, indent=2, separators=(',', ': ')))

        self.layers = []
        with variable_scope(type(self).__name__) as self.__varscope:

            prev_output, prev_output_shape = input_B_Di, input_shape

            for i_layer, ls in enumerate(layerspec):
                with variable_scope('layer_%d' % i_layer):
                    if ls['type'] == 'reshape':
                        _check_keys(ls, ['type', 'new_shape'], [])
                        self.layers.append(ReshapeLayer(prev_output, ls['new_shape']))

                    elif ls['type'] == 'fc':
                        _check_keys(ls, ['type', 'n'], ['initializer'])
                        self.layers.append(AffineLayer(
                            prev_output, prev_output_shape, output_shape=(ls['n'],), initializer=_parse_initializer(ls)))

                    elif ls['type'] == 'conv':
                        _check_keys(ls, ['type', 'chanout', 'filtsize', 'outsize', 'stride', 'padding'], ['initializer'])
                        self.layers.append(ConvLayer(
                            input_B_Ih_Iw_Ci=prev_output, input_shape=prev_output_shape,
                            Co=ls['chanout'],
                            Fh=ls['filtsize'], Fw=ls['filtsize'],
                            Oh=ls['outsize'], Ow=ls['outsize'],
                            Sh=ls['stride'], Sw=ls['stride'],
                            padding=ls['padding'],
                            initializer=_parse_initializer(ls)))

                    elif ls['type'] == 'nonlin':
                        _check_keys(ls, ['type', 'func'], [])
                        self.layers.append(NonlinearityLayer(prev_output, prev_output_shape, ls['func']))

                    else:
                        raise NotImplementedError('Unknown layer type %s' % ls['type'])

                prev_output, prev_output_shape = self.layers[-1].output, self.layers[-1].output_shape
                self._output, self._output_shape = prev_output, prev_output_shape

    @property
    def varscope(self): return self.__varscope
    @property
    def output(self): return self._output
    @property
    def output_shape(self): return self._output_shape

