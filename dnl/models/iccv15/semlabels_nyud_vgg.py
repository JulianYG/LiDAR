'''
Copyright (C) 2014 New York University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import os
import time
import numpy as np
import ipdb
import cPickle

import theano
import theano.tensor as T

from common import imgutil, logutil

import matplotlib.pyplot as plt

import thutil
from thutil import test_shape, theano_function, maximum, gpu_contiguous

from net import *
from pooling import cmrnorm, sum_unpool_2d
from utils import zero_pad_batch

from dataset_defs import NYUDepthModelDefs

_log = logutil.getLogger()
xx = np.newaxis

def read_label_map(fd, comment_chars='#%'):
    '''
    Reads label mapping file.
    Each line in the file contains "src label name = dst label name"
    Returns tuple (dst_label_names, label_mapping)
    where label_mapping maps a src label to a dst label index.
    '''
    srcname_to_dstname = {}
    for line in fd.readlines():
        line = line.strip()
        if not line or line[0] in comment_chars:
            continue
        (src, dst) = line.split('=')
        src = src.strip().lower()
        dst = dst.strip().lower()
        if src in srcname_to_dstname and dst != srcname_to_dstname[src]:
            raise ValueError('Label "%s" mapped to both "%s" and "%s"'
                             % (src, srcname_to_dstname[src], dst))
        srcname_to_dstname[src] = dst
    
    dst_label_names = sorted(set(srcname_to_dstname.values()))
    dst_label_names.insert(0, 'none')

    dstname_to_ind = dict(map(reversed, enumerate(dst_label_names)))
    srcname_to_dstind = {src: dstname_to_ind[dst]
                         for (src, dst) in srcname_to_dstname.iteritems()}

    return (dst_label_names, srcname_to_dstind)

class machine(Machine, NYUDepthModelDefs):
    def __init__(self, conf):
        self.define_meta()
        Machine.__init__(self, conf)

    def infer_labels(self, images, depths, normals):
        '''
        Infers semantic label maps maps for a list of 320x240 images,
        with RGB, Depth and Normals.
        returns a label map corresponding to the center box
        in the original rgbdn image.
        '''
        images = images.transpose((0,3,1,2))
        depths = depths.transpose((0,3,1,2))
        normals = normals.transpose((0,3,1,2))
        (nimgs, nc, nh, nw) = images.shape
        assert (nc, nh, nw) == (3,) + self.orig_input_size
        assert depths.shape == (nimgs, 1, nh, nw)
        assert normals.shape == (nimgs, 3, nh, nw)

        (input_h, input_w) = self.input_size
        (output_h, output_w) = self.output_size

        bsize = self.bsize
        b = 0

        # theano function for inference
        v = self.vars
        pred_labels = self.scale3.labels.pred_mean
        infer_f = theano.function([v.images, v.depths, v.normals],
                                  pred_labels)
        labels = np.zeros((nimgs, self.nlabels, output_h, output_w),
                          dtype=np.float32)

        # crop region (from random translations in training)
        dh = nh - input_h
        dw = nw - input_w
        (i0, i1) = (dh/2, nh - dh/2)
        (j0, j1) = (dw/2, nw - dw/2)

        # infer depth for images in batches
        b = 0
        while b < nimgs:
            batch_images = images[b:b+bsize]
            batch_depths = depths[b:b+bsize]
            batch_normals = normals[b:b+bsize]
            n = len(batch_images)
            if n < bsize:
                batch_images = zero_pad_batch(batch_images, bsize)
                batch_depths = zero_pad_batch(batch_depths, bsize)
                batch_normals = zero_pad_batch(batch_normals, bsize)

            # crop to network input size
            batch_images = batch_images[:, :, i0:i1, j0:j1]
            batch_depths = batch_depths[:, :, i0:i1, j0:j1]
            batch_normals = batch_normals[:, :, i0:i1, j0:j1]

            # infer depth with nnet
            batch_labels = infer_f(batch_images, batch_depths, batch_normals)
            labels[b:b+n] = batch_labels[:n]
            
            b += n

        return labels

    def get_predicted_region(self):
        '''
        Returns the region of a 320x240 image covered by the predicted
        depth map (y0 y1 x0 x1) where y runs the 240-dim and x runs the 320-dim.
        '''
        # found using trace-back of pixel inds through train target
        return (11, 227, 13, 305)

    def make_test_values(self):
        (input_h, input_w) = self.input_size
        (output_h, output_w) = self.output_size
        test_images_size = (self.bsize, 3, input_h, input_w)
        test_depths_size = (self.bsize, 1, input_h, input_w)
        test_normals_size = (self.bsize, 3, input_h, input_w)
        test_labels_size = (self.bsize, output_h, output_w)
        test_masks_size = (self.bsize, output_h, output_w)

        test_values = {}
        test_values['images'] = \
            (255 * np.random.rand(*test_images_size)).astype(np.float32)
        test_values['depths'] = \
            np.random.randn(*test_depths_size).astype(np.float32)
        test_values['normals'] = \
            np.random.randn(*test_normals_size).astype(np.float32)
        test_values['labels'] = \
            (41*np.random.rand(*test_labels_size)).astype(np.uint8)
        test_values['masks'] = \
            np.ones(test_masks_size, dtype=np.float32)
        return test_values

    def define_machine(self):
        self.scale2_size = self.conf.geteval('train', 'scale2_size')
        self.scale3_size = self.conf.geteval('train', 'scale3_size')
        (input_h, input_w) = self.input_size
        (scale2_h, scale2_w) = self.scale2_size
        (scale3_h, scale3_w) = self.scale3_size
        self.output_size = self.scale3_size

        self.use_depth_normals = \
                self.conf.getboolean('data', 'use_depth_normals')

        # 40-class label set
        output_label_names = self.conf.geteval('data', 'label_names')
        self.nlabels = len(output_label_names)

        # input vars
        images = T.tensor4('images')
        if self.use_depth_normals:
            depths = T.tensor4('depths')
            normals = T.tensor4('normals')
        labels_target = T.tensor3('labels_target', dtype='uint8')
        masks = T.tensor3('masks')

        test_values = self.make_test_values()
        images.tag.test_value = test_values['images']
        if self.use_depth_normals:
            depths.tag.test_value = test_values['depths']
            normals.tag.test_value = test_values['normals']
        masks.tag.test_value  = test_values['masks']
        labels_target.tag.test_value = test_values['labels']

        x0 = images
        if self.use_depth_normals:
            depths0 = depths
            normals0 = normals
        labels0 = labels_target

        self.inputs = MachinePart(locals())

        # downsample by 2x
        labels0 = labels0[:,1::2,1::2][:,:-1,:-1]

        # features for scale1 stack from imagenet
        self.define_imagenet_stack(x0)

        # pretrained features are rather large, rescale down to nicer range
        imnet_r5 = 0.01 * self.imagenet.r5
        imnet_feats = imnet_r5.reshape((
                            self.bsize, T.prod(imnet_r5.shape[1:])))

        # rest of scale1 stack
        self.define_scale1_stack(imnet_feats)

        # scale2 stack
        self.define_scale2_stack(x0, depths0, normals0)

        # scale3 stack
        self.define_scale3_stack(x0, depths0, normals0)

        self.vars = MachinePart(locals())

    def define_imagenet_stack(self, x0):
        conv1_1 = self.create_unit('imnet_conv1_1')
        conv1_2 = self.create_unit('imnet_conv1_2')
        pool1 = self.create_unit('imnet_pool1')

        conv2_1 = self.create_unit('imnet_conv2_1')
        conv2_2 = self.create_unit('imnet_conv2_2')
        pool2 = self.create_unit('imnet_pool2')

        conv3_1 = self.create_unit('imnet_conv3_1')
        conv3_2 = self.create_unit('imnet_conv3_2')
        conv3_3 = self.create_unit('imnet_conv3_3')
        pool3 = self.create_unit('imnet_pool3')

        conv4_1 = self.create_unit('imnet_conv4_1')
        conv4_2 = self.create_unit('imnet_conv4_2')
        conv4_3 = self.create_unit('imnet_conv4_3')
        pool4 = self.create_unit('imnet_pool4')

        conv5_1 = self.create_unit('imnet_conv5_1')
        conv5_2 = self.create_unit('imnet_conv5_2')
        conv5_3 = self.create_unit('imnet_conv5_3')
        pool5 = self.create_unit('imnet_pool5')

        x0 = x0 - self.meta.vgg_image_mean[xx,:,xx,xx]

        z1_1 = conv1_1.infer(x0)
        r1_1 = relu(z1_1)
        z1_2 = conv1_2.infer(r1_1)
        (p1, s1) = pool1.infer(z1_2)
        r1 = relu(p1)

        z2_1 = conv2_1.infer(r1)
        r2_1 = relu(z2_1)
        z2_2 = conv2_2.infer(r2_1)
        (p2, s2) = pool2.infer(z2_2)
        r2 = relu(p2)

        z3_1 = conv3_1.infer(r2)
        r3_1 = relu(z3_1)
        z3_2 = conv3_2.infer(r3_1)
        r3_2 = relu(z3_2)
        z3_3 = conv3_3.infer(r3_2)
        r3_3 = relu(z3_3)
        (p3, s3) = pool3.infer(z3_3)
        r3 = relu(p3)

        z4_1 = conv4_1.infer(r3)
        r4_1 = relu(z4_1)
        z4_2 = conv4_2.infer(r4_1)
        r4_2 = relu(z4_2)
        z4_3 = conv4_3.infer(r4_2)
        r4_3 = relu(z4_3)
        (p4, s4) = pool4.infer(z4_3)
        r4 = relu(p4)

        z5_1 = conv5_1.infer(r4)
        r5_1 = relu(z5_1)
        z5_2 = conv5_2.infer(r5_1)
        r5_2 = relu(z5_2)
        z5_3 = conv5_3.infer(r5_2)
        r5_3 = relu(z5_3)
        (p5, s5) = pool5.infer(z5_3)
        r5 = relu(p5)

        #r5_vec = r5.reshape((r5.shape[0], T.prod(r5.shape[1:])))
        #full6 = self.create_unit('imnet_full6',
        #                         ninput=test_shape(r5_vec)[1])
        #z6 = 0.5 * full6.infer(r5_vec)
        #r6 = relu(z6)

        #full7 = self.create_unit('imnet_full7', ninput=test_shape(r6)[1])
        #z7 = 0.5 * full7.infer(r6)
        #r7 = relu(z7)

        #full8 = self.create_unit('imnet_full8', ninput=test_shape(r7)[1])
        #z8 = full8.infer(r7)

        #output = softmax(z8, axis=1)

        self.imagenet = MachinePart(locals())

    def define_scale1_stack(self, imnet_feats):
        full1 = self.create_unit('labels_full1',
                                 ninput=test_shape(imnet_feats)[1])
        f_1 = relu(full1.infer(imnet_feats))
        f_1_drop = random_zero(f_1, 0.8)
        f_1_mean = 0.2 * f_1

        (fh, fw) = self.scale2_size
        full2 = self.create_unit('labels_full2',
                                 ninput=test_shape(f_1_mean)[1])
        
        f_2_drop = full2.infer(f_1_drop)
        f_2_mean = full2.infer(f_1_mean)

        (fh, fw) = self.scale2_size
        full2_feature_size = self.conf.geteval('labels_full2', 'feature_size')
        (nfeat, nh, nw) = full2_feature_size
        assert (nh, nw) == (14, 19) and (fh, fw) == (55, 74)

        # upsample feature maps to scale2 size
        f_2_drop = f_2_drop.reshape((self.bsize, nfeat, nh, nw))
        f_2_mean = f_2_mean.reshape((self.bsize, nfeat, nh, nw))
        f_2_drop_up = upsample_bilinear(f_2_drop, 4)[:, :, 2:-2, 2:-3]
        f_2_mean_up = upsample_bilinear(f_2_mean, 4)[:, :, 2:-2, 2:-3]
        assert test_shape(f_2_drop_up)[-2:] == (fh, fw)

        self.scale1 = MachinePart(locals())

    def define_scale2_stack(self, x0, depths0, normals0):
        # input features
        x0_pproc = (x0 - self.meta.images_mean) \
                   * self.meta.images_istd

        if self.use_depth_normals:
            assert self.conf.get('data', 'depth_space') == 'log'
            depths0_pproc = T.log(depths0 + 1e-4)

        conv_s2_1_rgb = self.create_unit('conv_s2_1_rgb')
        z_s2_1_rgb = conv_s2_1_rgb.infer(x0_pproc)

        conv_s2_1_d = self.create_unit('conv_s2_1_d')
        z_s2_1_d = conv_s2_1_d.infer(depths0_pproc)

        conv_s2_1_n = self.create_unit('conv_s2_1_n')
        z_s2_1_n = conv_s2_1_n.infer(normals0)

        pool_s2_1 = self.create_unit('pool_s2_1')
        (p_s2_1_rgb, _) = pool_s2_1.infer(z_s2_1_rgb)
        (p_s2_1_d, _) = pool_s2_1.infer(z_s2_1_d)
        (p_s2_1_n, _) = pool_s2_1.infer(z_s2_1_n)

        p_s2_1 = T.concatenate((p_s2_1_rgb, p_s2_1_d, p_s2_1_n), axis=1)
        p_s2_1 = relu(p_s2_1)

        # concat input features
        p_1_drop = T.concatenate((self.scale1.f_2_drop_up,
                                  p_s2_1),
                                 axis=1)
        p_1_mean = T.concatenate((self.scale1.f_2_mean_up,
                                  p_s2_1),
                                 axis=1)

        # conv stack from first-layer features of scale2
        labels = self.define_scale2_onestack('labels', p_1_drop, p_1_mean)
        self.scale2 = MachinePart(locals())

    def define_scale2_onestack(self, stack_type, p_1_drop, p_1_mean):
        conv_s2_2 = self.create_unit('%s_conv_s2_2' % stack_type)
        z_s2_2_drop    = relu(conv_s2_2.infer(p_1_drop))
        z_s2_2_mean    = relu(conv_s2_2.infer(p_1_mean))

        conv_s2_3 = self.create_unit('%s_conv_s2_3' % stack_type)
        z_s2_3_drop    = relu(conv_s2_3.infer(z_s2_2_drop))
        z_s2_3_mean    = relu(conv_s2_3.infer(z_s2_2_mean))

        conv_s2_4 = self.create_unit('%s_conv_s2_4' % stack_type)
        z_s2_4_drop    = relu(conv_s2_4.infer(z_s2_3_drop))
        z_s2_4_mean    = relu(conv_s2_4.infer(z_s2_3_mean))

        conv_s2_5 = self.create_unit('%s_conv_s2_5' % stack_type)
        z_s2_5_drop    = conv_s2_5.infer(z_s2_4_drop)
        z_s2_5_mean    = conv_s2_5.infer(z_s2_4_mean)

        # prediction
        pred_drop = z_s2_5_drop
        pred_mean = z_s2_5_mean

        assert stack_type == 'labels'
        # pixelwise softmax directly from outputs (small number of classes)
        pred_drop = softmax(pred_drop[:, :self.nlabels, :, :], axis=1)
        pred_mean = softmax(pred_mean[:, :self.nlabels, :, :], axis=1)

        return MachinePart(locals())

    def define_scale3_training_crop(self, output_size, crop_size):
        (oh, ow) = output_size
        (ch, cw) = crop_size
        rh = T.floor(theano_rng.uniform() * (oh - ch)).astype('int32')
        rw = T.floor(theano_rng.uniform() * (ow - cw)).astype('int32')
        rh.tag.test_value = np.int32(0.2 * (oh - ch))
        rw.tag.test_value = np.int32(0.2 * (ow - cw))
        target_crop_h = slice(rh, rh+ch)
        target_crop_w = slice(rw, rw+cw)
        input_crop_h = slice(2*rh, 2*(rh+ch+1)+8)
        input_crop_w = slice(2*rw, 2*(rw+cw+1)+8)
        return (target_crop_h, target_crop_w, input_crop_h, input_crop_w)

    def define_scale3_stack(self, x0, depths0, normals0):
        # input features
        (fh, fw) = self.scale2_size
        (target_crop_h, target_crop_w, input_crop_h, input_crop_w) = \
                self.define_scale3_training_crop((2*fh-1, 2*fw-1),
                                                 (fh, fw))

        x0_crop = x0[:, :, input_crop_h, input_crop_w]
        if self.use_depth_normals:
            depths0_crop = depths0[:, :, input_crop_h, input_crop_w]
            normals0_crop = normals0[:, :, input_crop_h, input_crop_w]

        x0_pproc = (x0 - self.meta.images_mean) \
                   * self.meta.images_istd

        x0_pproc_crop = (x0_crop - self.meta.images_mean) \
                        * self.meta.images_istd

        if self.use_depth_normals:
            assert self.conf.get('data', 'depth_space') == 'log'
            depths0_pproc = T.log(depths0 + 1e-4)
            depths0_pproc_crop = T.log(depths0_crop + 1e-4)

        conv_s3_1_rgb = self.create_unit('conv_s3_1_rgb')
        z_s3_1_rgb = conv_s3_1_rgb.infer(x0_pproc)
        z_s3_1_rgb_crop = conv_s3_1_rgb.infer(gpu_contiguous(x0_pproc_crop))

        conv_s3_1_d = self.create_unit('conv_s3_1_d')
        z_s3_1_d = conv_s3_1_d.infer(depths0_pproc)
        z_s3_1_d_crop = conv_s3_1_d.infer(gpu_contiguous(depths0_pproc_crop))

        conv_s3_1_n = self.create_unit('conv_s3_1_n')
        z_s3_1_n = conv_s3_1_n.infer(normals0)
        z_s3_1_n_crop = conv_s3_1_n.infer(gpu_contiguous(normals0_crop))

        pool_s3_1 = self.create_unit('pool_s3_1')
        (p_s3_1_rgb, _) = pool_s3_1.infer(z_s3_1_rgb)
        (p_s3_1_d, _) = pool_s3_1.infer(z_s3_1_d)
        (p_s3_1_n, _) = pool_s3_1.infer(z_s3_1_n)
        (p_s3_1_rgb_crop, _) = pool_s3_1.infer(z_s3_1_rgb_crop)
        (p_s3_1_d_crop, _) = pool_s3_1.infer(z_s3_1_d_crop)
        (p_s3_1_n_crop, _) = pool_s3_1.infer(z_s3_1_n_crop)

        p_s3_1 = T.concatenate((p_s3_1_rgb, p_s3_1_d, p_s3_1_n), axis=1)
        p_s3_1 = relu(p_s3_1)

        p_s3_1_crop = T.concatenate((p_s3_1_rgb_crop,
                                     p_s3_1_d_crop,
                                     p_s3_1_n_crop),
                                    axis=1)
        p_s3_1_crop = relu(p_s3_1_crop)

        def _upsamp_2_to_3(x):
            return thutil.constant(upsample_constant(x, 2)[:,:,:-1,:-1])

        # concat input features with scale2 prediction
        p_1_drop = T.concatenate(
                        (_upsamp_2_to_3(self.scale2.labels.pred_drop)
                            [:, :, target_crop_h, target_crop_w],
                         p_s3_1_crop),
                        axis=1)
        p_1_mean = T.concatenate(
                        (_upsamp_2_to_3(self.scale2.labels.pred_mean),
                         p_s3_1),
                        axis=1)

        labels = self.define_scale3_onestack(
                            'labels', p_1_drop, p_1_mean)

        self.scale3 = MachinePart(locals())

    def define_scale3_onestack(self, stack_type, p_1_drop, p_1_mean):
        conv_s3_2 = self.create_unit('%s_conv_s3_2' % stack_type)
        conv_s3_3 = self.create_unit('%s_conv_s3_3' % stack_type)
        conv_s3_4 = self.create_unit('%s_conv_s3_4' % stack_type)

        z_s3_2_drop    = relu(conv_s3_2.infer(p_1_drop))
        z_s3_2_mean    = relu(conv_s3_2.infer(p_1_mean))

        z_s3_3_drop    = relu(conv_s3_3.infer(z_s3_2_drop))
        z_s3_3_mean    = relu(conv_s3_3.infer(z_s3_2_mean))

        z_s3_4_drop    = conv_s3_4.infer(z_s3_3_drop)
        z_s3_4_mean    = conv_s3_4.infer(z_s3_3_mean)

        # prediction
        pred_drop = z_s3_4_drop
        pred_mean = z_s3_4_mean

        assert stack_type == 'labels'
        # pixelwise softmax
        pred_drop = softmax(pred_drop[:, :self.nlabels, :, :], axis=1)
        pred_mean = softmax(pred_mean[:, :self.nlabels, :, :], axis=1)

        return MachinePart(locals())


    def define_labels_cost(self, pred, labels0):
        npix = T.prod(pred.shape[-2:])
        valid = T.neq(labels0, 0)[:,xx,:,:].astype('float32')
        labels_bin = T.eye(self.nlabels)[labels0].transpose((0,3,1,2))

        # cross entropy
        labels_error = -T.sum(valid * labels_bin * T.log(pred)) / T.sum(valid)
        labels_cost = labels_error

        return (labels_error, labels_cost)

