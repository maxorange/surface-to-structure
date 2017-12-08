import tensorflow as tf
from ops import *
import cv2
import os
import nets
import utils

class SurfaceToStructure(Model):

    def __init__(self, args, sess=None):
        self.session(sess)
        self.G_losses = []
        self.D_losses = []
        self.xgs = []
        self.netE = nets.Encoder()
        self.netG = nets.Generator()
        self.netD = nets.Discriminator()
        self.train_G = tf.placeholder(tf.bool)
        self.train_D = tf.placeholder(tf.bool)
        opt = tf.train.AdamOptimizer(args.learning_rate, 0.5)
        G_tower_grads = []
        D_tower_grads = []
        n_disc = len(args.nsf_disc)

        for i in range(args.n_gpus):
            gpu_name = '/gpu:{0}'.format(i+3)
            with tf.device(gpu_name):
                print gpu_name
                batch_size_per_gpu = args.batch_size / args.n_gpus
                self.build_network(args, batch_size_per_gpu, i)

                G_grads = opt.compute_gradients(self.G_losses[-1], var_list=self.E_vars + self.G_vars)
                G_tower_grads.append(G_grads)

                D_grads = [opt.compute_gradients(self.D_losses[-1][i], var_list=self.D_vars[i]) for i in range(n_disc)]
                D_tower_grads.append(D_grads)

        self.optG = opt.apply_gradients(average_gradients(G_tower_grads))
        self.G_loss = tf.reduce_mean(self.G_losses)
        self.xg = tf.concat(self.xgs, 0)

        self.optD = []
        self.D_loss = []
        for i in range(n_disc):
            grads = []
            losses = []
            for j in range(args.n_gpus):
                grads.append(D_tower_grads[j][i])
                losses.append(self.D_losses[j][i])
            self.optD.append(opt.apply_gradients(average_gradients(grads)))
            self.D_loss.append(tf.reduce_mean(losses))

        if sess is None and args.check == False:
            self.initialize()

        ma_vars = tf.moving_average_variables()
        BN_vars = [var for var in ma_vars if var.name.startswith('E') or var.name.startswith('G')]
        variables_to_save = self.E_vars + self.G_vars + BN_vars
        super(SurfaceToStructure, self).__init__(variables_to_save)

    def build_network(self, args, batch_size, gpu_idx):
        reuse = False if gpu_idx == 0 else True
        x = tf.placeholder(tf.float32, [batch_size, args.npx, args.npx, 1], 'x'+str(gpu_idx))
        y = tf.placeholder(tf.float32, [batch_size, args.npx, args.npx, 1], 'y'+str(gpu_idx))

        # generator networks
        enc_hs = self.netE(y, args.nsf_gen, args.npx, self.train_G, reuse=reuse)
        xg = self.netG(enc_hs, args.nsf_gen, args.npx, self.train_G, reuse=reuse)
        self.xgs.append(xg)

        # discriminator networks
        n_disc = len(args.nsf_disc)
        logits_real = []
        logits_fake = []
        for i in range(n_disc):
            logits_real.append(self.netD(x, y, args.nsf_disc[i], args.npx, self.train_D, 'D'+str(i), reuse=reuse))
            logits_fake.append(self.netD(xg, y, args.nsf_disc[i], args.npx, self.train_D, 'D'+str(i), reuse=True))

        # collect trainable variables
        if gpu_idx == 0:
            t_vars = tf.trainable_variables()
            self.E_vars = [var for var in t_vars if var.name.startswith('E')]
            self.G_vars = [var for var in t_vars if var.name.startswith('G')]
            self.D_vars = []
            for i in range(n_disc):
                self.D_vars.append([var for var in t_vars if var.name.startswith('D'+str(i))])

        # generator loss
        G_losses_adv = [[tf.reduce_mean(sigmoid_kl_with_logits(logits_fake[i], 0.8))] for i in range(n_disc)]
        G_loss_adv = weighted_arithmetic_mean(args.adv_weights, tf.concat(G_losses_adv, 0))
        G_loss_rec = tf.reduce_mean(tf.abs(x - xg))
        G_weight_decay = tf.add_n([tf.nn.l2_loss(var) for var in self.E_vars + self.G_vars])
        self.G_losses.append(G_loss_adv + 100*G_loss_rec + 5e-4*G_weight_decay)

        # discriminator loss
        D_loss = []
        for i in range(n_disc):
            D_loss_real = tf.reduce_mean(sigmoid_kl_with_logits(logits_real[i], 0.8))
            D_loss_fake = tf.reduce_mean(sigmoid_ce_with_logits(logits_fake[i], tf.zeros_like(logits_fake[i])))
            D_weight_decay = tf.add_n([tf.nn.l2_loss(var) for var in self.D_vars[i]])
            D_loss.append(D_loss_real + D_loss_fake + 5e-4*D_weight_decay)
        self.D_losses.append(D_loss)

    def optimize_generator(self, x, y):
        fd = {self.train_G:True, self.train_D:True}
        for i, v in enumerate(x): fd['x{0}:0'.format(i)] = v
        for i, v in enumerate(y): fd['y{0}:0'.format(i)] = v
        self.sess.run(self.optG, feed_dict=fd)

    def optimize_discriminator(self, x, y):
        fd = {self.train_G:True, self.train_D:True}
        for i, v in enumerate(x): fd['x{0}:0'.format(i)] = v
        for i, v in enumerate(y): fd['y{0}:0'.format(i)] = v
        self.sess.run(self.optD, feed_dict=fd)

    def get_errors(self, x, y):
        fd = {self.train_G:False, self.train_D:False}
        for i, v in enumerate(x): fd['x{0}:0'.format(i)] = v
        for i, v in enumerate(y): fd['y{0}:0'.format(i)] = v
        D_loss = self.sess.run(self.D_loss, feed_dict=fd)
        G_loss = self.sess.run(self.G_loss, feed_dict=fd)
        return D_loss, G_loss

    def generate(self, y):
        fd = {self.train_G: False}
        for i, v in enumerate(y): fd['y{0}:0'.format(i)] = v
        return self.sess.run(self.xg, feed_dict=fd)

    def save_log(self, x, y, epoch, batch, out_path):
        D_loss, G_loss = self.get_errors(x, y)
        xg = self.generate(y)

        # save generated samples
        for i, l in enumerate(xg):
            img = utils.tanh2uint16(l)
            filename = os.path.join(out_path, '{0}-{1}.png'.format(epoch, i))
            cv2.imwrite(filename, img)

        # write error rates to log_file
        with open(self.log_file, 'a') as f:
            ld = ', '.join(['{0:.8f}'.format(l) for l in D_loss])
            print >> f, '{0:>3}, {1:>5}, {2}, {3:.8f}'.format(epoch, batch, ld, G_loss)

    def run(self, args, dataset):
        params_path = os.path.join('params', args.version)
        out_path = os.path.join('out', args.version)
        if not os.path.exists(params_path): os.mkdir(params_path)
        if not os.path.exists(out_path): os.mkdir(out_path)
        self.create_log_file(os.path.join(out_path, 'log.txt'))

        total_batch = dataset.num_examples / args.batch_size
        x_test, y_test = dataset.read_test_data(args.batch_size)

        for epoch in range(1, args.n_iters+1):
            for batch in range(total_batch):
                # update discriminator
                x, y = dataset.next_batch(args.batch_size)
                self.optimize_discriminator(x, y)

                # update generator
                x, y = dataset.next_batch(args.batch_size)
                self.optimize_generator(x, y)

                if batch % args.log_interval == 0:
                    self.save_log(x_test, y_test, epoch, batch, out_path)

            if epoch % args.save_interval == 0:
                filename = os.path.join(params_path, 'epoch-{0}.ckpt'.format(epoch))
                self.save(filename)
