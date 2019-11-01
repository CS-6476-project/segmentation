import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io
import numpy as np

image_base_name = '196027.jpg'

feature_spaces = ['hsv', 'hsv_pos', 'rgb', 'rgb_pos']

root_dir = '..'
data_dir = os.path.join(root_dir, 'BSDS500', 'BSDS500', 'data')
segs_dir = os.path.join(data_dir, 'segs')

km_output_dir = os.path.join(segs_dir, 'k_means')
ms_output_dir = os.path.join(segs_dir, 'mean_shift')
groud_truth_dir = os.path.join(data_dir, 'groundTruth', 'test')
test_img_dir = os.path.join(data_dir, 'images', 'test')

image_name = os.path.splitext(image_base_name)[0]

for directory in [ms_output_dir, km_output_dir]:
  for fs in feature_spaces:
    mat = loadmat(os.path.join(directory, fs, '%s.mat' % image_name))
    # hopefully you don't have more than 256 clusters...
    segs = np.uint8(mat['segs'][0, 0]) 

    dir_base = os.path.basename(directory)
    plt.imsave('%s_%s_%s.png' % (dir_base, fs, image_name), segs)

# Save any one groundTruth segments
groundTruth = loadmat(os.path.join(groud_truth_dir, '%s.mat' % image_name))
groundTruthSegs = groundTruth['groundTruth']
to_pick = np.random.randint(groundTruthSegs.shape[1])
groundTruthSegs = groundTruthSegs[0, to_pick][0, 0]
groundTruthSegs = np.uint8(groundTruthSegs[0])
plt.imsave('ground_truth_%s.png' % image_name, groundTruthSegs)

# Save the original image here as well for convenience 
plt.imsave('original_%s.png' % image_name, io.imread(os.path.join(test_img_dir, image_base_name)))
