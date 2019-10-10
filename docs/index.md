---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

# Project Proposal

## Problem statement

The goal of our project is to compare different segmentation methods that we have learned about in class with current state-of-the-art techniques. We will also explore how different feature spaces affect clustering and graph-based approaches to segmentation.

The expected input to our system will be images from our chosen dataset. The desired output will be a graph/some visualization depicting and analyzing which segmentation methods worked better. There will also be a visual comparison between techniques learned in class and current state-of-the-art techniques.

## Approach

The techniques we will be comparing include:

1. Clustering-based approaches:

   - K-means clustering
   - Mean shift clustering

2. Graph-based approaches:

   - Min-cut
   - Normalized cut

3. State-of-the-art, Mask R-CNN

We will use the following feature spaces:

1. RGB Color
2. RGB Color + Position
3. HSV color
4. HSV color + Position
5. Texture using the Gabor filter bank

For each feature space, k-means will be run multiple times with different centroid seeds and the final result will be the best output of the consecutive runs in terms of Sum of Squared Error. The initialized centroids will be generally distant from each other, and we will use Euclidean distance as our distance function.

The bandwidth parameter for mean shift clustering will be estimated by the library code, and the number of modes found by mean shift will be used as the value of k in k-means.

Graph-based approaches will also be run for each feature space. Affinity weight for edges between pairs of pixels will be given by an exponential similarity function of Euclidean distance in that feature space. The min-cut approach will employ Edmonds-Karp or any other standard max-flow algorithm. For normalized cut, an approximate solution will be computed by solving the generalized eigenvalue problem on the affinity matrix.

We will treat the Mask R-CNN architecture and algorithm as a black box and make only minor modifications to suit our dataset’s needs.

## Experiments and results

We will be using the BSDS500 (Berkeley Segmentation) dataset [1]. This dataset consists of 500 color images, which are split into 200 training, 100 validation, and 200 test images. Each segmentation method will be evaluated on this dataset.

We will use the following existing code:

- Texture: Garbor kernel implementation in the skimage library.
- K-means clustering: Scikit-learn’s implementation of the k-means algorithm.
- Mean shift clustering: Scikit-learn’s implementation of the mean shift algorithm. Scikit-learn also provides a utility to estimate the bandwidth parameter of the RBF kernel used.
- Min-cut: Standard Python implementation of Edmonds-Karp algorithm readily available online.
- Normalized cut: Skimage’s implementation which recursively performs a 2-way normalized cut on a passed in Region Adjacency Graph.
- Mask R-CNN: A convenient implementation of Mask R-CNN, with trained weights on the COCO dataset provided [here.](https://github.com/matterport/Mask_RCNN)

We will write our own code for the following:

- Mapping pixels to feature spaces. Utility functions to convert between color spaces will help but we will have to write most of the code ourselves.
- Finding the textons, the texton histogram and then clustering the histograms.
- Our own configuration and dataset parsing script to run the Mask R-CNN code on the BSDS500 dataset. Since this dataset doesn’t have enough examples to train a deep learning model from scratch, we will have to take a transfer learning approach and use weights trained from the COCO dataset.
- Code to compute metrics and quantitatively evaluate the quality of our segmentation. We will also aim to show interesting visualizations comparing the results from different methods.

We will evaluate the segmentation algorithms in two ways:

1. Find the segment boundaries and evaluate boundary benchmarks like the F-measure.
2. Evaluate the quality of segment regions using clustering metrics such as Rand Index, Variation of Information and Segmentation Covering [2].

We expect the experiment results to favor the Mask R-CNN approach, but we are curious to see how well classical CV techniques perform in comparison. It is expected that normalized-cut will perform better than min-cut since min-cut tends to pull out small isolated components. Amongst the clustering-based approaches, mean-shift should edge out k-means since it doesn’t make any assumptions about the data distribution. It will be interesting to see how different feature spaces impact each of the algorithms.

## Citation

[1] D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural
images and its application to evaluating segmentation algorithms and measuring eco-
logical statistics. In Proc. 8th Int’l Conf. Computer Vision, volume 2, pages 416–423,
July 2001.

[2] Arbelaez, Pablo & Maire, Michael & Fowlkes, Charless & Malik, Jitendra. (2011). Contour Detection and Hierarchical Image Segmentation. IEEE transactions on pattern analysis and machine intelligence. 33. 898-916. 10.1109/TPAMI.2010.161.

# Team Members

- Anand Chaturvedi, achaturvedi32
- Prabhav Chawla, pchawla8
- Pranshav Thakkar, pthakkar7
- Sanskriti Rathi, srathi7
