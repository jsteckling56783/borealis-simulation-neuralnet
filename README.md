# borealis-simulation-neuralnet
PCA of Aurora Borealis images for dimensionality reduction

Upon completion: generative video output using LTSM deep learning

### What is Principal Component Analysis (PCA)?
PCA describes a dataset, often of images, using a set of common features called principal components. 
For example, in images of faces this could include distance between eyes, mouth width, ect.

In this dataset, features could express shapes of bursts of light in certain locations.
Check out this visualization: https://youtu.be/4VAkrUNLKSo?t=187

![Dimentionality reduction results: 10 dimensions](pcaDemo/10DimensionReduction.png)
![PCA study of cumulative component importance](pcaDemo/pcaFeatures.png)
![Dimentionality reduction results: 120 dimensions](pcaDemo/120DimensionReduction.png)
![Dimentionality reduction results: 120 dimensions, enlarged](pcaDemo/120DimensionReductionEnlarged.png)
![Dimentionality reduction results: 300 dimensions](pcaDemo/300DimensionReduction.png)
![Images naively generated with uniform PCA values](pcaDemo/generatedRandom100pca.png)
![Comparison of distribution of top two PCA features](pcaDemo/NormalFeatureCompaison.png)



# Naive sequence generation (resulting images in GeneratedImagesRandomDistrSequence):
For each principal component, generae a random number in the uniform distribution [0,1] and map it
to the corresponding value along the trained normal distribution for that PC on the input data. 
Note the flaw in this method: images hover around 1 similar image which represents the mean weights.
This is a motivating factor to use LSTM.

![Naive sequence generation step 0](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_0.png)
![Naive sequence generation step 1](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_1.png)
![Naive sequence generation step 2](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_2.png)
![Naive sequence generation step 3](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_3.png)
![Naive sequence generation step 4](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_4.png)
![Naive sequence generation step 5](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_5.png)
![Naive sequence generation step 6](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_6.png)
![Naive sequence generation step 7](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_7.png)
![Naive sequence generation step 8](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_8.png)
![Naive sequence generation step 9](GeneratedImagesRandomDistrSequence/genSeqImgsOrd_9.png)

At each time step, move the latent value [0,1] by a slight, randomly generated amount. This causes it to move 
slightly along the normal curve. Incorrectly colored images using a random vector of initial PCA weights instead of
a sample from the normal distrbution of that PC's weights from training.

![Naive sequence generation, without respect to trained distribution of Principal Component weights step 0](GeneratedImagesRandomDistrSequence/genSeqImgsLg_0.png)
![Naive sequence generation step 1](GeneratedImagesRandomDistrSequence/genSeqImgsLg_1.png)
![Naive sequence generation step 2](GeneratedImagesRandomDistrSequence/genSeqImgsLg_2.png)
![Naive sequence generation step 3](GeneratedImagesRandomDistrSequence/genSeqImgsLg_3.png)
![Naive sequence generation step 4](GeneratedImagesRandomDistrSequence/genSeqImgsLg_4.png)
![Naive sequence generation step 5](GeneratedImagesRandomDistrSequence/genSeqImgsLg_5.png)
![Naive sequence generation step 6](GeneratedImagesRandomDistrSequence/genSeqImgsLg_6.png)
![Naive sequence generation step 7](GeneratedImagesRandomDistrSequence/genSeqImgsLg_7.png)
![Naive sequence generation step 8](GeneratedImagesRandomDistrSequence/genSeqImgsLg_8.png)
![Naive sequence generation step 9](GeneratedImagesRandomDistrSequence/genSeqImgsLg_9.png)

The above examples have been color corrected for extreme color values that do not occur in
the original set and manifest themselves as outliers (i.e. bright red). 
These are the images without color correction

![Naive sequence generation, without respect to trained distribution of Principal Component weights step 0](GeneratedImagesRandomDistrSequence/genSeqImgs_0.png)
![Naive sequence generation step 1](GeneratedImagesRandomDistrSequence/genSeqImgs_1.png)
![Naive sequence generation step 2](GeneratedImagesRandomDistrSequence/genSeqImgs_2.png)
![Naive sequence generation step 3](GeneratedImagesRandomDistrSequence/genSeqImgs_3.png)
![Naive sequence generation step 4](GeneratedImagesRandomDistrSequence/genSeqImgs_4.png)
![Naive sequence generation step 5](GeneratedImagesRandomDistrSequence/genSeqImgs_5.png)
![Naive sequence generation step 6](GeneratedImagesRandomDistrSequence/genSeqImgs_6.png)
![Naive sequence generation step 7](GeneratedImagesRandomDistrSequence/genSeqImgs_7.png)
![Naive sequence generation step 8](GeneratedImagesRandomDistrSequence/genSeqImgs_8.png)
![Naive sequence generation step 9](GeneratedImagesRandomDistrSequence/genSeqImgs_9.png)