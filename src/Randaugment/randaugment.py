from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
"""
This repository is build upon RandAugment implementation
https://arxiv.org/abs/1909.13719 published here
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py

"""
#Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AutoAugment and RandAugment policies for enhanced image preprocessing.
AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""
import inspect
import numpy as np
import math
from tensorboard.plugins.hparams import api as hp
from PIL import Image, ImageEnhance, ImageOps
from Randaugment.augmenters.color.hedcoloraugmenter import HedColorAugmenter
import random
import albumentations as A

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

def hed(image, factor):
    factor = random.randint(0, 20)/100
    #print('image',image.shape)
    image=np.transpose(image,[2,0,1])
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=(-factor, factor), haematoxylin_bias_range=(-factor, factor),
                                            eosin_sigma_range=(-factor, factor), eosin_bias_range=(-factor, factor),
                                            dab_sigma_range=(-factor, factor), dab_bias_range=(-factor, factor),
                                            cutoff_range=(0.15, 0.85))
    ##To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image),[1,2,0])
    

NAME_TO_FUNC = {
    'Hed': hed
  }

def _enhance_level_to_arg_hed(level):
  return (level*0.03,)
  

def level_to_arg(hparams):
  return {
      'Hed': _enhance_level_to_arg_hed
  }


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams,magnitude):
  """Return the function that corresponds to `name` and update `level` param."""
  if name=='Hed':
      func = NAME_TO_FUNC[name]
      args = level_to_arg(augmentation_hparams)[name](magnitude)
  
  # Check to see if prob is passed into function. This is used for operations
  # where we alter bboxes independently.
  # pytype:disable=wrong-arg-types
  if 'prob' in inspect.getargspec(func)[0]:
    args = tuple([prob] + list(args))
  # pytype:enable=wrong-arg-types

  # Add in replace arg if it is required for the function that is being called.
  # pytype:disable=wrong-arg-types
  if 'replace' in inspect.getargspec(func)[0]:
    # Make sure replace is the final argument
    assert 'replace' == inspect.getargspec(func)[0][-1]
    args = tuple(list(args) + [replace_value])
  # pytype:enable=wrong-arg-types

  return (func, prob, args)



def distort_image_with_randaugment(image, num_layers, magnitude,ra_type):
  """Applies the RandAugment policy to `image`.
  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [1, 10].
    ra_type: List of augmentations to use
  Returns:
    The augmented version of `image`.
  """
  #print(magnitude)
  replace_value = (128, 128, 128)#[128] * 3
  #tf.logging.info('Using RandAug.')
  augmentation_hparams = {'cutout_const':40, 'translate_const':10}
  #The 'Default' option is the H&E tailored randaugment
  if ra_type== 'Default': 
    available_ops = ['Hed']  
  for layer_num in range(num_layers):
    op_to_select = np.random.randint(low=0,high=len(available_ops))
    random_magnitude = np.random.uniform(low=0, high=magnitude)
    for (i, op_name) in enumerate(available_ops):
        prob = np.random.uniform(low=0.2, high=0.8)
        func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                           replace_value, augmentation_hparams,magnitude)

        if  (i== op_to_select):
            selected_func=func
            selected_args=args
            image= selected_func(image, *selected_args)
            
        else: 
            image=image

  return image
