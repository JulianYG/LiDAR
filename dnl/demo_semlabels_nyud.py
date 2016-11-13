import os
import sys
import numpy as np
import cPickle
from PIL import Image

import net
from utils import multichannel_montage

def unpack_depths(depths_float16):
    return depths_float16.astype(np.float32)

def unpack_normals(normals_uint8):
    normals = normals_uint8.astype(np.float32)
    normals *= (2.0 / 255.0)
    normals -= 1
    normals /= np.sqrt(np.maximum(np.sum(normals**2, axis=2, keepdims=True),
                                  1e-6))
    return normals

def main():
    # location of depth module, config and parameters
        
    model_name = 'semlabels_nyud4_vgg'   # 4-class model
    #model_name = 'semlabels_nyud14_vgg'  # 14-class model
    #model_name = 'semlabels_nyud40_vgg'  # 40-class model

    module_fn = 'models/iccv15/semlabels_nyud_vgg.py'
    config_fn = 'models/iccv15/%s.conf' % model_name
    params_dir = 'weights/iccv15/%s' % model_name

    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)

    # demo image
    sample_image = cPickle.load(open('demo_nyud_rgbdn_image.pk'))
    image = sample_image['image']
    depths = unpack_depths(sample_image['depths'])
    normals = unpack_normals(sample_image['normals'])

    # build depth inference function and run
    image = image[np.newaxis, :, :, :]
    depths = depths[np.newaxis, :, :, np.newaxis]
    normals = normals[np.newaxis, :, :, :]
    pred_labels = machine.infer_labels(image, depths, normals)

    # save prediction
    labels_img_np = multichannel_montage(pred_labels)
    labels_img = Image.fromarray((255*labels_img_np).astype(np.uint8))
    labels_img.save('demo_labels_prediction.png')


if __name__ == '__main__':
    main()


