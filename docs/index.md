---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Team SegFault
---

<link rel="stylesheet" href="assets/css/custom.css">

# Final Project Update

## Abstract

The goal of our project is to compare different segmentation methods that we have learned about in class with current state-of-the-art techniques. We will also explore how different feature spaces affect clustering and graph-based approaches to segmentation.

**TODO** @Anand _Edit:_

~~For this milestone, we have written code to parse the Berkeley Segmentation image dataset (BSDS500) to extract various feature spaces from images, to run the K-Means and Mean Shift algorithms and to evaluate our segments using standard clustering metrics. According to our preliminary results, Mean Shift outperforms K-Means, and the HSV + Position feature space shows the most promising results.~~

## Teaser Figure

<div class="resultsContainer">
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

**TODO** @Anand _Edit or Delete:_

~~So far, we have been able to assess K-Means and Mean Shift’s performance on the test images. Using the ground truth, we have calculated region and boundary benchmarks, which are discussed in the Results section.~~

## Approach

**TODO** @Anand _Edit the midterm update content and merge content written by others to make this section cohesive._

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

An obstacle we faced was figuring out how to convert our segmented images into a format that can be interpreted by the benchmarking code. This was especially challenging since we wrote our code in Python, but the benchmarking code has been written in MATLAB.

**TODO** @Sanskriti _Mention libraries used to implement Normalized Cut. Any obstacles faced? Any design choices, or judgment calls?_

For the state-of-the-art deep learning approach, we chose the Context Encoding Network (EncNet) implementation by H. Zhang et al. (2018) [3]. The training code and pre-trained models are readily available on [GitHub](https://github.com/zhanghang1989/PyTorch-Encoding). We decided against taking a transfer learning approach since the BSDS500 dataset lacks class labels, so tuning a larger, pre-trained EncNet on BSDS500 was not possible without a lot of manual annotation work. In some sense, this is good test of EncNet's ability to generalize to a novel dataset that it has not seen before.

## Experiments and Results

**TODO** @Anand _Split the following appropriately between the Mean Shift and K-Means sub-sections. Also complete the two sub-sections by adding any missing content._

First, we focused on running the Mean Shift algorithm since the results would be used to determine 'k' values for K-Means.

We followed the following experimental set-up:

1. For each test image, compute a representation in all of the four feature spaces. Obtain an array of data for each one of those feature spaces.
2. Normalize the arrays - each feature vector has a mean of 0 and a standard deviation of 1 after the normalization. Clustering-based approaches generally benefit from data normalization.
3. For each array, estimate the bandwidth and run Mean Shift. Collect the clustering labels and the total number of clusters generated.
4. Save the number of clusters, for each image and feature space, in a dictionary for later use by K-Means.
5. Assign clustering labels to image pixels (basically, perform a mapping from feature space to image space). Save segmented images to disk in a format expected by the benchmarking code.

### Mean Shift

**TODO** @Anand _See TODO above_

### K-Means

The process described above was repeated for K-Means (leaving out certain Mean Shift specific steps). Sample [output](https://cs-6476-project.herokuapp.com/?q=235098) for all feature spaces is shown below:

<div class="resultsContainer">
  <div class="resultImageContainer">
    <img src="assets/roller_coaster_235098/original.png" />
    Original image
  </div>
  <div class="resultImageContainer">
    <img src="assets/roller_coaster_235098/ground_truth_num=21.png" />
    <span>Ground truth, <span class="italic">segs=21</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/roller_coaster_235098/k_means_rgb_num=7.png" />
    <span>RGB space, <span class="italic">segs=7</span></span>
  </div>
    <div class="resultImageContainer">
    <img src="assets/roller_coaster_235098/k_means_hsv_num=7.png" />
    <span>HSV space, <span class="italic">segs=7</span></span>
  </div>
    <div class="resultImageContainer">
    <img src="assets/roller_coaster_235098/k_means_rgb_pos_num=8.png" />
    <span>RGB + Pos space, <span class="italic">segs=9</span></span>
  </div>
    <div class="resultImageContainer">
    <img src="assets/roller_coaster_235098/k_means_hsv_pos_num=9.png" />
    <span>HSV + Pos space, <span class="italic">segs=9</span></span>
  </div>
</div>

For this particular image, all feature spaces do a reasonably good job. The impact of adding position is very apparent based on the output above - the sky is segmented into more parts when pixel position is taken into account.

### Normalized Cut

**TODO** @Sanskriti

### EncNet_ResNet101_PContext

We treated the EncNet implementation as a black box and did not make any modifications to the existing codebase. The pre-trained model uses the ResNet-101 convolutional neural network as a basic feature extractor and the PASCAL-Context dataset for training, which is a standard semantic segmentation dataset consisting of about 30k images and 400+ labels. The model made predictions on raw, test image data, and was not fed hand-crafted, feature-space specific features. Here is an [illustration](https://cs-6476-project.herokuapp.com/?q=80085) of the output we obtain:

<div class="resultsContainer">
  <div class="resultImageContainer">
    <img src="assets/lady_cloth_80085/original.jpg" />
    Original image
  </div>
  <div class="resultImageContainer">
    <img src="assets/lady_cloth_80085/ground_truth_num=28.png" />
    <span>Ground truth, <span class="italic">segs=28</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/lady_cloth_80085/encnet_num=12.png" />
    <span>EncNet, <span class="italic">segs=12</span></span>
  </div>
</div>

The EncNet approach is very good at segmenting individual objects from the background scene; however, it scarcely divides up different parts that make up an individual. In the ground truth image above, the lady in the foreground is divided into at least four different segments but EncNet uses just one segment. This is not necessarily a bad thing, it really depends on a case-by-case basis whether individuals should be represented by more than one segment.

### Quantitative Results

We ran our benchmarking code on a total of 2600 segmented images (200 test images × 4 feature spaces/image × 3 approaches with feature spaces + 200 test images × 1 approach without feature spaces) and 200 ground truth segments.

**TODO** @Prabhav _Add/Edit content related to final update. Don't forget the graph!_

The following graph summarizes the F-measure metric:

<div id="graphContainer">
  <img src="assets/f_measure_graph.png" id="graph"/>
</div>

<!--
| Segmentation Approach           | F-measure |
| ------------------------------- | :-------: |
| K-Means, RGB space              |   0.38    |
| Mean Shift, RGB space           |   0.45    |
| Normalized Cut, RGB space       |   0.58    |
| K-Means, HSV space              |   0.41    |
| Mean Shift, HSV space           |   0.49    |
| Normalized Cut, HSV space       |   0.56    |
| K-Means, RGB + Pos space        |   0.44    |
| Mean Shift, RGB + Pos space     |   0.48    |
| Normalized Cut, RGB + Pos space |   0.49    |
| K-Means, HSV + Pos space        |   0.46    |
| Mean Shift, HSV + Pos space     |   0.50    |
| Normalized Cut, HSV + Pos space |   0.51    |
| EncNet                          |   0.44    |
-->

The following table summarizes our region-based metrics:

| Segmentation Approach           | PRI  | VOI  | Covering |
| ------------------------------- | :--: | ---- | -------- |
| K-Means, RGB space              | 0.69 | 3.32 | 0.32     |
| Mean Shift, RGB space           | 0.67 | 2.88 | 0.39     |
| Normalized Cut, RGB space       | 0.75 | 2.19 | 0.50     |
| K-Means, HSV space              | 0.69 | 3.00 | 0.36     |
| Mean Shift, HSV space           | 0.61 | 2.47 | 0.41     |
| Normalized Cut, HSV space       | 0.75 | 2.16 | 0.51     |
| K-Means, RGB + Pos space        | 0.71 | 2.63 | 0.36     |
| Mean Shift, RGB + Pos space     | 0.70 | 2.60 | 0.40     |
| Normalized Cut, RGB + Pos space | 0.75 | 2.68 | 0.44     |
| K-Means, HSV + Pos space        | 0.73 | 2.79 | 0.37     |
| Mean Shift, HSV + Pos space     | 0.70 | 2.41 | 0.44     |
| Normalized Cut, HSV + Pos space | 0.76 | 2.65 | 0.45     |
| EncNet                          | 0.74 | 2.06 | 0.52     |

Given that a human performs with F = 0.79, our vanilla implementations of simple clustering algorithms don’t perform too bad (Mean Shift obtains F = 0.50).

Mean Shift, across feature spaces, has (marginally) better F-measure values than K-Means. We also see higher Probabilistic Rand Index and lower Variation of Information with Mean Shift.

As far as feature spaces are concerned, HSV + Position achieves the best metrics, and RGB + Position. This makes sense as we can take into account more information with these feature spaces.

## Qualitative Results

We illustrate our results for a sample image:

<div class="resultsContainer">
  <div class="resultImageContainer">
    <img src="assets/couple_145059/original.png" class="largerHeight" />
    Original image
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/ground_truth_num=26.png" class="largerHeight" />
    <span>Ground truth, <span class="italic">segs=26</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/encnet_num=3.png" class="largerHeight" />
    <span>EncNet, <span class="italic">segs=3</span></span>
  </div> 
  <div class="resultImageContainer">
    <img src="assets/couple_145059/k_means_rgb_num=7.png" class="largerHeight" />
    <span>K-Means, RGB space, <span class="italic">segs=7</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/mean_shift_rgb_num=7.png" class="largerHeight" />
    <span>Mean Shift, RGB space, <span class="italic">segs=7</span></span>
  </div>

  <div class="resultImageContainer">
    <img src="assets/couple_145059/normalized_cut_rgb_num=71.png" class="largerHeight" />
    <span>Normalized Cut, RGB space, <span class="italic">segs=71</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/k_means_hsv_num=7.png" class="largerHeight" />
    <span>K-Means, HSV space, <span class="italic">segs=7</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/mean_shift_hsv_num=7.png" class="largerHeight" />
    <span>Mean Shift, HSV space, <span class="italic">segs=7</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/normalized_cut_hsv_num=51.png" class="largerHeight" />
    <span>Normalized Cut, HSV space, <span class="italic">segs=51</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/k_means_rgb_pos_num=5.png" class="largerHeight" />
    <span>K-Means, RGB + Pos space, <span class="italic">segs=5</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/mean_shift_rgb_pos_num=5.png" class="largerHeight" />
    <span>Mean Shift, RGB + Pos space, <span class="italic">segs=5</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/normalized_cut_rgb_pos_num=67.png" class="largerHeight" />
    <span>Normalized Cut, RGB + Pos space, <span class="italic">segs=67</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/k_means_hsv_pos_num=8.png" class="largerHeight" />
    <span>K-Means, HSV + Pos space, <span class="italic">segs=8</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/mean_shift_hsv_pos_num=8.png" class="largerHeight" />
    <span>Mean Shift, HSV + Pos space, <span class="italic">segs=8</span></span>
  </div>
  <div class="resultImageContainer">
    <img src="assets/couple_145059/normalized_cut_hsv_pos_num=72.png" class="largerHeight" />
    <span>Normalized Cut, HSV + Pos space, <span class="italic">segs=72</span></span>
  </div>
</div>

To view more results, please visit the website: <https://cs-6476-project.herokuapp.com>. On every page load, the website randomly picks an image from the test set and displays the corresponding segmented images.

## Conclusion and Futurework

**TODO** @Sanskriti _Add/Edit content related to final update. "In the Conclusions section, you should re-iterate the goals that you had laid out in your mid-term update and clearly mention whether you were able to accomplish those goals. If not, then what were the reasons. Also, in the Future Work section, you should mention any interesting extensions of your project that you can think of."_

~~In the coming weeks, we would like to repeat the same process for two graph-based and a deep-learning, state-of-the-art approaches. We will also aim to incorporate texture as a feature space in computing our results.~~

## References

[1] D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural
images and its application to evaluating segmentation algorithms and measuring eco-
logical statistics. In Proc. 8th Int’l Conf. Computer Vision, volume 2, pages 416–423,
July 2001.

[2] Arbelaez, Pablo & Maire, Michael & Fowlkes, Charless & Malik, Jitendra. (2011). Contour Detection and Hierarchical Image Segmentation. IEEE transactions on pattern analysis and machine intelligence. 33. 898-916. doi: 10.1109/TPAMI.2010.161.

[3] Zhang, H., Dana, K., Shi, J., Zhang, Z., Wang, X., Tyagi, A., & Agrawal, A. (2018). Context Encoding for Semantic Segmentation. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. doi: 10.1109/cvpr.2018.00747

# Team Members

- Anand Chaturvedi, achaturvedi32
- Prabhav Chawla, pchawla8
- Pranshav Thakkar, pthakkar7
- Sanskriti Rathi, srathi7

### [Link to project proposal](./proposal.md)

### [Link to midterm update](./midterm.md)
