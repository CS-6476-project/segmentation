import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io
# This code is not that flexible. Use https://cs-6476-project.herokuapp.com/?q=FILE_NAME instead
import numpy as np

image_base_name = '235098.jpg'

feature_spaces = ['hsv', 'hsv_pos', 'rgb', 'rgb_pos']

root_dir = '..'
data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
segs_dir = os.path.join(data_dir, 'segs')

segs_dir_names = ['k_means', 'mean_shift', 'normalized_cut']
groud_truth_dir = os.path.join(data_dir, 'groundTruth', 'test')
test_img_dir = os.path.join(data_dir, 'images', 'test')

image_name = os.path.splitext(image_base_name)[0]

output_dir = os.path.join('.', image_name)
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

for directory in segs_dir_names:
  for fs in feature_spaces:
    mat = loadmat(os.path.join(segs_dir, directory, fs, '%s.mat' % image_name))
    # hopefully you don't have more than 256 clusters...
    segs = np.uint8(mat['segs'][0, 0])
    num_segs = np.unique(segs).size

    dir_base = os.path.basename(directory)
    plt.imsave(os.path.join(output_dir, '%s_%s_num=%d.png' % (dir_base, fs, num_segs)), segs)

# Save any one groundTruth segments
groundTruth = loadmat(os.path.join(groud_truth_dir, '%s.mat' % image_name))
groundTruthSegs = groundTruth['groundTruth']
to_pick = np.random.randint(groundTruthSegs.shape[1])
groundTruthSegs = groundTruthSegs[0, to_pick][0, 0]
groundTruthSegs = np.uint8(groundTruthSegs[0])
num_segs = np.unique(groundTruthSegs).size
plt.imsave(os.path.join(output_dir, 'ground_truth_num=%d.png' % num_segs), groundTruthSegs)

# Save the original image here as well for convenience
plt.imsave(os.path.join(output_dir, 'original.png'), io.imread(os.path.join(test_img_dir, image_base_name)))
