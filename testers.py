import tensorflow as tf
from ops import *
import nets

class SurfaceToStructure(Model):

    def __init__(self, args, sess=None):
        self.session(sess)
        self.netE = nets.Encoder()
        self.netG = nets.Generator()
        self.train = tf.placeholder(tf.bool)
        self.build_network(args.nsf, args.npx, args.batch_size)

        if sess is None and args.check == False:
            self.initialize()

        variables_to_restore = tf.trainable_variables() + tf.moving_average_variables()
        super(SurfaceToStructure, self).__init__(variables_to_restore)

    def build_network(self, nsf, npx, batch_size):
        self.y = tf.placeholder(tf.float32, [batch_size, npx, npx, 1], 'y')
        enc_hs = self.netE(self.y, nsf, npx, self.train)
        self.xg = self.netG(enc_hs, nsf, npx, self.train)

    def generate(self, y):
        fd = {self.y:y, self.train:False}
        return self.sess.run(self.xg, feed_dict=fd)
