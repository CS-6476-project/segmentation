---
layout: default
title: Mid-Term Update
---

<link rel="stylesheet" href="assets/css/custom.css">

# Mid-Term Project Update

## Abstract

The goal of our project is to compare different segmentation methods that we have learned about in class with current state-of-the-art techniques. We will also explore how different feature spaces affect clustering and graph-based approaches to segmentation.

For this milestone, we have written code to parse the Berkeley Segmentation image dataset (BSDS500) to extract various feature spaces from images, to run the K-Means and Mean Shift algorithms and to evaluate our segments using standard clustering metrics. According to our preliminary results, Mean Shift outperforms K-Means, and the HSV + Position feature space shows the most promising results.

## Teaser Figure

<div id="teaserContainer">
  <img src="assets/teaser_196027/original.png" />
  <img src="assets/teaser_196027/ground_truth.png" /> 
  <img src="assets/teaser_196027/k_means_rgb.png" /> 
  <img src="assets/teaser_196027/mean_shift_rgb.png" /> 
  <img src="assets/teaser_196027/k_means_hsv_pos.png" /> 
  <img src="assets/teaser_196027/mean_shift_hsv_pos.png" /> 
</div>

## Introduction

The motivation behind our project is to compare different segmentation techniques and evaluate how they are affected by different feature spaces. The final objective is to see how methods learned in class compare against state-of-the-art methods for segmentation.

The domain for our project consists of regular RGB images, taken from the BSDS500 dataset. The dataset contains 200 training, 100 validation, and 200 test images and human annotations for all these images, which serve as the ground truth segmentations. The BSDS500 is an industry-standard for evaluating segmentation and contour detection algorithms. [1]

So far, we have been able to assess K-Means and Mean Shift’s performance on the test images. Using the ground truth, we have calculated region and boundary benchmarks, which are discussed in the Results section.

## Approach

We ran K-Means and Mean Shift algorithms on the following four feature spaces:

1. RGB color
2. RGB color + Position (x, y pixel coordinates)
3. HSV color
4. HSV color + Position (x, y pixel coordinates)

We used Scikit-learn’s implementation of K-Means and Mean Shift. The bandwidth parameter for Mean Shift clustering was estimated using utility functions provided by Scikit-learn.

One drawback of using K-Means is that one has to specify the number of clusters. To avoid blindly trying different ‘k’ values for each image and each feature space, we used the number of modes picked up by Mean Shift as the value of ‘k’ for that particular image and that feature space. Instead of manually looking at each image, and guessing the number of clusters for each feature space of that image, we were able to automate the process and save a great deal of time and computer resources. We believe this choice is justified since it represents domain knowledge injection into our problem.

We used the benchmarking code provided by the BSDS500 dataset maintainers to obtain our metrics. This ensured that our metrics were reliably computed and were in a format that is easy to compare with existing benchmarks. The following metrics were computed for each algorithm and feature space:

1. F-measure; used to evaluate segment boundaries. [2]
2. Probabilistic Rand Index (PRI), Variation of Information (VOI) and Segmentation Covering; region-based metrics used to evaluate the quality of segments. [2]

Each metric was averaged across the 200 test images and reported twice - once while choosing an optimal scale for the entire dataset (ODS) and the other when selecting an optimal scale per image (OIS). For simplicity's sake, we only report the OIS measures here.

An obstacle we faced was figuring out how to convert our segmented images into a format that can be interpreted by the benchmarking code. This was especially challenging since we wrote our code in Python, but the benchmarking code has been written in MATLAB.

## Experiments and Results

First, we focused on running the Mean Shift algorithm since the results would be used to determine 'k' values for K-Means.

We followed the following experimental set-up:

1. For each test image, compute a representation in all of the four feature spaces. Obtain an array of data for each one of those feature spaces.
2. Normalize the arrays - each feature vector has a mean of 0 and a standard deviation of 1 after the normalization. Clustering-based approaches generally benefit from data normalization.
3. For each array, estimate the bandwidth and run Mean Shift. Collect the clustering labels and the total number of clusters generated.
4. Save the number of clusters, for each image and feature space, in a dictionary for later use by K-Means.
5. Assign clustering labels to image pixels (basically, perform a mapping from feature space to image space). Save segmented images to disk in a format expected by the benchmarking code.

The above process was repeated for K-Means (leaving out certain Mean Shift specific steps). In total, we had 1600 segmented images (200 images × 4 feature spaces/image × 2 clustering algorithms) and 200 ground truth segments to run our benchmarking code.

The following table summarizes the F-measure metric:

| Segmentation Approach            | F-measure |
| -------------------------------- | :-------: |
| K-Means, RGB space               |   0.38    |
| Mean Shift, RGB space            |   0.45    |
| K-Means, HSV space               |   0.41    |
| Mean Shift, HSV space            |   0.49    |
| K-Means, RGB + Position space    |   0.44    |
| Mean Shift, RGB + Position space |   0.48    |
| K-Means, HSV + Position space    |   0.46    |
| Mean Shift, HSV + Position space |   0.50    |

The following table summarizes our region-based metrics:

| Segmentation Approach            | PRI  | VOI  | Covering |
| -------------------------------- | :--: | ---- | -------- |
| K-Means, RGB space               | 0.69 | 3.32 | 0.32     |
| Mean Shift, RGB space            | 0.67 | 2.88 | 0.39     |
| K-Means, HSV space               | 0.69 | 3.00 | 0.36     |
| Mean Shift, HSV space            | 0.61 | 2.47 | 0.41     |
| K-Means, RGB + Position space    | 0.71 | 2.63 | 0.36     |
| Mean Shift, RGB + Position space | 0.70 | 2.60 | 0.40     |
| K-Means, HSV + Position space    | 0.73 | 2.79 | 0.37     |
| Mean Shift, HSV + Position space | 0.70 | 2.41 | 0.44     |

Given that a human performs with F = 0.79, our vanilla implementations of simple clustering algorithms don’t perform too bad (Mean Shift obtains F = 0.50).

Mean Shift, across feature spaces, has (marginally) better F-measure values than K-Means. We also see higher Probabilistic Rand Index and lower Variation of Information with Mean Shift.

As far as feature spaces are concerned, HSV + Position achieves the best metrics, and RGB + Position. This makes sense as we can take into account more information with these feature spaces.

## Qualitative Results

We illustrate our results for a couple of images:

<div id="resultsContainer">
  <div class="resultsColumn">
    <div class="resultImageContainer"><img src="assets/couple_145059/original.png" />(a)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/ground_truth_num=26.png" />(b)</div> 
    <div class="resultImageContainer"><img src="assets/couple_145059/k_means_rgb_num=7.png" />(c)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/mean_shift_rgb_num=7.png" />(d)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/k_means_hsv_num=7.png" />(e)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/mean_shift_hsv_num=7.png" />(f)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/k_means_rgb_pos_num=5.png" />(g)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/mean_shift_rgb_pos_num=5.png" />(h)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/k_means_hsv_pos_num=8.png" />(i)</div>
    <div class="resultImageContainer"><img src="assets/couple_145059/mean_shift_hsv_pos_num=8.png" />(j)</div>
  </div>
  <div class="resultsColumn">
    <div class="resultImageContainer"><img src="assets/taj_288024/original.png" />(a)</div>
    <div class="resultImageContainer"><img src="assets/taj_288024/ground_truth_num=32.png" />(b)</div>
    <div class="resultImageContainer"><img src="assets/taj_288024/k_means_rgb_num=4.png" />(c)</div>
    <div class="resultImageContainer"><img src="assets/taj_288024/mean_shift_rgb_num=4.png" />(d)</div>
    <div class="resultImageContainer"><img src="assets/taj_288024/k_means_hsv_num=3.png" />(e)</div>
    <div class="resultImageContainer"><img src="assets/taj_288024/mean_shift_hsv_num=3.png" />(f)</div>
    <div class="resultImageContainer"><img src="assets/taj_288024/k_means_rgb_pos_num=5.png" />(g)</div> 
    <div class="resultImageContainer"><img src="assets/taj_288024/mean_shift_rgb_pos_num=5.png" />(h)</div> 
    <div class="resultImageContainer"><img src="assets/taj_288024/k_means_hsv_pos_num=4.png" />(i)</div> 
    <div class="resultImageContainer"><img src="assets/taj_288024/mean_shift_hsv_pos_num=4.png" />(j)</div>
  </div>
</div>

For each collection of images, top-to-bottom labeling:

<ol id="resultLabels">
  <li>Original Image</li>
  <li>Ground truth segmentation</li>
  <li>K-Means, RGB space segmentation</li>
  <li>Mean Shift, RGB space segmentation</li>
  <li>K-Means, HSV space segmentation</li>
  <li>Mean Shift, HSV space segmentation</li>
  <li>K-Means, RGB + Position space segmentation</li>
  <li>Mean Shift, RGB + Position space segmentation</li>
  <li>K-Means, HSV + Position space segmentation</li>
  <li>Mean Shift, HSV + Position space segmentation</li>
</ol>

## Conclusion and Futurework

In the coming weeks, we would like to repeat the same process for two graph-based and a deep-learning, state-of-the-art approaches. We will also aim to incorporate texture as a feature space in computing our results.

## References

[1] D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural
images and its application to evaluating segmentation algorithms and measuring eco-
logical statistics. In Proc. 8th Int’l Conf. Computer Vision, volume 2, pages 416–423,
July 2001.

[2] Arbelaez, Pablo & Maire, Michael & Fowlkes, Charless & Malik, Jitendra. (2011). Contour Detection and Hierarchical Image Segmentation. IEEE transactions on pattern analysis and machine intelligence. 33. 898-916. 10.1109/TPAMI.2010.161.
