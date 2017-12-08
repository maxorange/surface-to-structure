import tensorflow as tf
from ops import *

class Encoder(object):

    def __call__(self, y, nsf, npx, train, name='E', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size, _, _, nc = y.get_shape().as_list()
            nf = 64 # number of filters
            layer_idx = 1
            enc_hs = []

            u = conv2d(y, [4, 4, nc, nf], 'h{0}'.format(layer_idx), stride=1)
            enc_hs.append(leaky_relu(batch_norm(u, train, 'bn{0}'.format(layer_idx))))

            while nsf < npx:
                layer_idx += 1
                u = conv2d(enc_hs[-1], [4, 4, nf, min(nf*2, 512)], 'h{0}'.format(layer_idx))
                enc_hs.append(leaky_relu(batch_norm(u, train, 'bn{0}'.format(layer_idx))))
                _, _, npx, nf = enc_hs[-1].get_shape().as_list()

            for h in enc_hs:
                print name, h.get_shape()
            return enc_hs

class Generator(object):

    def __call__(self, enc_hs, nsf, npx, train, name='G', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size, _, _, nf = enc_hs[-1].get_shape().as_list()
            layer_idx = 1
            h = enc_hs[-layer_idx]

            u = deconv2d(h, [4, 4, nf, nf], [batch_size, nsf*2, nsf*2, nf], 'h{0}'.format(layer_idx))
            h = tf.nn.relu(batch_norm(u, train, 'bn{0}'.format(layer_idx)))
            _, _, nsf, nf = h.get_shape().as_list()
            print name, h.get_shape()

            while nsf < npx:
                layer_idx += 1
                h0 = enc_hs[-layer_idx]
                h1 = enc_hs[-layer_idx-1]
                if h0.get_shape()[-1] != h1.get_shape()[-1]:
                    out_ch = nf/2
                else:
                    out_ch = nf
                c = tf.concat([h, h0], -1)
                u = deconv2d(c, [4, 4, out_ch, nf*2], [batch_size, nsf*2, nsf*2, out_ch], 'h{0}'.format(layer_idx))
                h = tf.nn.relu(batch_norm(u, train, 'bn{0}'.format(layer_idx)))
                _, _, nsf, nf = h.get_shape().as_list()
                print name, h.get_shape()

            layer_idx += 1
            xg = deconv2d(h, [4, 4, 1, nf], [batch_size, npx, npx, 1], 'h{0}'.format(layer_idx), bias=True, stride=1)
            print name, xg.get_shape()
            return tf.tanh(xg)

class Discriminator(object):

    def __call__(self, x, y, nsf, npx, train, name='D', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            nf = 128 # number of filters
            layer_idx = 1

            xy = tf.concat([x, y], -1)
            batch_size, _, _, nc = xy.get_shape().as_list()

            u = conv2d(xy, [4, 4, nc, nf], 'h{0}'.format(layer_idx), bias=True)
            h = leaky_relu(u)
            _, _, npx, nf = h.get_shape().as_list()
            print name, h.get_shape()

            while nsf < npx:
                layer_idx += 1
                u = conv2d(h, [4, 4, nf, min(nf*2, 512)], 'h{0}'.format(layer_idx))
                h = leaky_relu(batch_norm(u, train, 'bn{0}'.format(layer_idx)))
                _, _, npx, nf = h.get_shape().as_list()
                print name, h.get_shape()

            layer_idx += 1
            padded_h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
            u = conv2d(padded_h, [4, 4, nf, min(nf*2, 512)], 'h{0}'.format(layer_idx), stride=1, padding='VALID')
            h = leaky_relu(batch_norm(u, train, 'bn{0}'.format(layer_idx)))
            _, _, npx, nf = h.get_shape().as_list()
            print name, h.get_shape()

            layer_idx += 1
            padded_h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
            logits = conv2d(padded_h, [4, 4, nf, 1], 'h{0}'.format(layer_idx), bias=True, stride=1, padding='VALID')
            print name, logits.get_shape()
            return logits
