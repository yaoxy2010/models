# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIZE = 300

# We want to crop an area that is at least MIN percentage of the image area,
# but not more than MAX percentage of the image area.
_MIN_CROP_RATIO = .40
_MAX_CROP_RATIO = .85
# When evaluating, use the mean as the chosen crop size.
_MEAN_CROP_RATIO = _MIN_CROP_RATIO / _MAX_CROP_RATIO


def _get_h_w(image_buffer):
  """Convenience function for getting float 32 height and width from the image.
  """
  image_shape = tf.image.extract_jpeg_shape(image_buffer)
  height = tf.cast(image_shape[0], tf.float32)
  width = tf.cast(image_shape[0], tf.float32)
  return height, width


def _random_crop_and_flip(image):
  """Crops the given image to a random part of the image, and randomly flips.

  Args:
    image: a 3-D image tensor

  Returns:
    3-D tensor with cropped image.

  """
  height, width = _get_h_w(image)

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  height_fraction = tf.random_uniform(
      [], minval=_MIN_CROP_RATIO, maxval=_MAX_CROP_RATIO, dtype=tf.float32)
  crop_height = height * height_fraction
  offset_y = height - crop_height

  width_fraction = tf.random_uniform(
      [], minval=_MIN_CROP_RATIO, maxval=_MAX_CROP_RATIO, dtype=tf.float32)
  crop_width = width * width_fraction
  offset_x = width - crop_width

  crop_window = tf.stack([offset_y, offset_x, crop_height, crop_width])

  # Results in a 3-D int8 Tensor. This will be converted to a float later,
  # during resizing.
  cropped = tf.image.decode_and_crop_jpeg(image, crop_window, channels=3)

  cropped = tf.image.random_flip_left_right(cropped)
  return cropped


def _central_crop(image):
  """Performs central crops of the given image.

  Args:
    image: a 3-D image tensor

  Returns:
    3-D tensor with cropped image.
  """
  height, width = _get_h_w(image)

  crop_height = height * _MEAN_CROP_RATIO
  offset_y = (height - crop_height) // 2

  crop_width = width * _MEAN_CROP_RATIO
  offset_x = (width - crop_width) // 2

  crop_window = tf.stack([offset_y, offset_x, crop_height, crop_width])

  # Results in a 3-D int8 Tensor. This will be converted to a float later,
  # during resizing.
  return tf.image.decode_and_crop_jpeg(image, crop_window, channels=3)


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  means = tf.expand_dims(tf.expand_dims(means, 0), 0)

  return image - means


def preprocess_image(image, output_height, output_width, is_training=False):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  if is_training:
    image = _random_crop_and_flip(image)
  else:
    image = _central_crop(image)

  # Note that resizing converts the image to float32
  image = tf.image.resize_images(
      image,
      [output_height, output_width],
      method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)

  num_channels = image.get_shape().as_list()[-1]
  image.set_shape([output_height, output_width, num_channels])

  return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
