import os
import numpy as np
from glob import glob
from skimage import io
from skimage.color import rgb2hsv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat
import pickle


def main():
  for index, img_path in enumerate(glob(os.path.join(test_img_dir, "*.jpg")), 1):
    print("\nImage %d" % index)
    img_base_name = os.path.basename(img_path)
    print("Running code for %s" % img_base_name)

    im = io.imread(img_path)
    hsv_im = rgb2hsv(im)

    quantizeHSV(hsv_im, img_base_name, max(2, hsv_modes_dict[img_base_name]))
    quantizeHSVAndPos(hsv_im, img_base_name, max(2, hsv_pos_modes_dict[img_base_name]))


def normalize_and_shuffle_data(data):
  data = normalizer.fit_transform(data)
  perm = np.arange(data.shape[0])
  np.random.shuffle(perm)
  data = data[perm]
  return data, np.argsort(perm)


def run_k_means(data, k):
  km = KMeans(n_clusters=k, random_state=0).fit(data)

  return km.labels_


def save_as_mat_file(data, img_name, output_dir):
  obj_arr = np.empty((1,), dtype=np.object)
  obj_arr[0] = data
  savemat(os.path.join(output_dir, "%s.mat" % img_name), {segs_str: obj_arr})


def quantizeHSVAndPos(hsv_im, img_base_name, k):
  height, width, channels = hsv_im.shape
  # Technically should be height first followed by width, but doesn't matter
  x, y = np.meshgrid(np.arange(width), np.arange(height))
  hsv_im_with_pos = np.concatenate([hsv_im, x[..., None], y[..., None]], axis=2)

  hsv_pos_space_img = np.reshape(hsv_im_with_pos, (height * width, channels + 2))
  hsv_pos_space_img, inv_perm = normalize_and_shuffle_data(hsv_pos_space_img)

  labels = run_k_means(hsv_pos_space_img, k)

  labels = labels[inv_perm]
  labels = np.reshape(labels, (height, width))

  img_name = os.path.splitext(img_base_name)[0]
  save_as_mat_file(labels, img_name, hsv_pos_output_dir)


def quantizeHSV(hsv_im, img_base_name, k):
  height, width, channels = hsv_im.shape
  hsv_space_img = np.reshape(hsv_im, (height * width, channels))
  hsv_space_img, inv_perm = normalize_and_shuffle_data(hsv_space_img)

  labels = run_k_means(hsv_space_img, k)

  labels = labels[inv_perm]
  labels = np.reshape(labels, (height, width))

  img_name = os.path.splitext(img_base_name)[0]
  save_as_mat_file(labels, img_name, hsv_output_dir)


if __name__ == '__main__':

  segs_str = 'segs'
  hsv_str = 'hsv'
  hsv_pos_str = 'hsv_pos'

  root_dir = '..'

  data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
  segs_dir = os.path.join(data_dir, segs_str)

  km_output_dir = os.path.join(segs_dir, 'k_means')
  test_img_dir = os.path.join(data_dir, 'images', 'test')

  hsv_output_dir = os.path.join(km_output_dir, hsv_str)
  hsv_pos_output_dir = os.path.join(km_output_dir, hsv_pos_str)

  for directory in [hsv_output_dir, hsv_pos_output_dir]:
    if not os.path.exists(directory):
      os.makedirs(directory)

  hsv_modes_dict = pickle.load(open(os.path.join(segs_dir, 'hsv_modes_dict.pickle'), 'rb'))
  hsv_pos_modes_dict = pickle.load(open(os.path.join(segs_dir, 'hsv_pos_modes_dict.pickle'), 'rb'))
  normalizer = StandardScaler()

  main()
