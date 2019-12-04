import os
import numpy as np
from glob import glob
from skimage import io, segmentation
from skimage.color import rgb2hsv
from skimage.future import graph
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat


def main():
  for index, img_path in enumerate(glob(os.path.join(test_img_dir, "*.jpg")), 1):
    print("\nImage %d" % index)
    img_base_name = os.path.basename(img_path)
    print("Running code for %s" % img_base_name)

    im = io.imread(img_path)
    hsv_im = rgb2hsv(im)

    # quantizeWithPos(im, img_base_name, rgb_pos_output_dir)
    # quantizeWithPos(hsv_im, img_base_name, hsv_pos_output_dir)
    quantizeWithoutPos(im, img_base_name, rgb_output_dir)
    hsv_im_scaled = np.uint8(hsv_im * 255)
    quantizeWithoutPos(hsv_im_scaled, img_base_name, hsv_output_dir)


def normalize_data(data):
  data = normalizer.fit_transform(data)
  return np.uint8(data)  # since normalizer has range [0, 255]


def run_k_means(im, compactness=0.01, n_segments=1000):
  return segmentation.slic(im, compactness=compactness, n_segments=n_segments)


def save_as_mat_file(data, img_name, output_dir):
  obj_arr = np.empty((1,), dtype=np.object)
  obj_arr[0] = data
  savemat(os.path.join(output_dir, "%s.mat" % img_name), {segs_str: obj_arr})


def quantizeWithoutPos(im, img_base_name, output_dir):
  labels1 = run_k_means(im, compactness=30, n_segments=400)
  g = graph.rag_mean_color(im, labels1, mode='similarity')
  labels2 = graph.cut_normalized(labels1, g)

  img_name = os.path.splitext(img_base_name)[0]
  save_as_mat_file(labels2, img_name, output_dir)


def quantizeWithPos(im, img_base_name, output_dir):
  height, width, channels = im.shape
  x, y = np.meshgrid(np.arange(width), np.arange(height))
  im_with_pos = np.concatenate([im, x[..., None], y[..., None]], axis=2)

  reshaped_im = np.reshape(im_with_pos, (height * width, channels + 2))
  normalized_im = normalize_data(reshaped_im)

  im_with_pos = np.reshape(normalized_im, (height, width, channels + 2))
  labels1 = run_k_means(im_with_pos)
  # NOTE: For this code to work correctly, change definition of 'total color' in rag.py to np.array([0, 0, 0, 0, 0])
  g = graph.rag_mean_color(im_with_pos, labels1, mode='similarity')
  labels2 = graph.cut_normalized(labels1, g)

  img_name = os.path.splitext(img_base_name)[0]
  save_as_mat_file(labels2, img_name, output_dir)


if __name__ == '__main__':

  segs_str = 'segs'
  hsv_str = 'hsv'
  hsv_pos_str = 'hsv_pos'
  rgb_str = 'rgb'
  rgb_pos_str = 'rgb_pos'

  root_dir = '..'

  data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
  segs_dir = os.path.join(data_dir, segs_str)

  nc_output_dir = os.path.join(segs_dir, 'normalized_cut')
  test_img_dir = os.path.join(data_dir, 'images', 'test')

  rgb_output_dir = os.path.join(nc_output_dir, rgb_str)
  rgb_pos_output_dir = os.path.join(nc_output_dir, rgb_pos_str)
  hsv_output_dir = os.path.join(nc_output_dir, hsv_str)
  hsv_pos_output_dir = os.path.join(nc_output_dir, hsv_pos_str)

  for directory in [rgb_output_dir, rgb_pos_output_dir, hsv_output_dir, hsv_pos_output_dir]:
    if not os.path.exists(directory):
      os.makedirs(directory)

  normalizer = MinMaxScaler(feature_range=(0, 255))

  main()
