## Global and Local Threshold

- **Global** thresholding consists of setting an intensity value (threshold) such that all voxels having intensity value below the threshold belong to one phase, the remainder belong to the other. 

- Global thresholding is as good as the degree of intensity separation between the two peaks in the image. It is an unsophisticated segmentation choice.

- In global threshold, we put the threshold equal 127. Then, we get the image is larger than the threshold.

  <img src="Images\Global_threshold.png" style="zoom:85%;" />

- **Local** thresholding is used to separate desirable foreground image objects from the background based on the difference in pixel intensities of each region.

- Local thresholding selects an individual threshold for each pixel based on the range of intensity values in its local neighborhood. This allows for thresholding of an image whose global intensity histogram doesn't contain distinctive peaks.

  <img src="Images\local_threshold.png" style="zoom:80%;" />

## Transformation

#### Gray Scale Image

- **grayscale** or image is one in which the value of each pixel is a single sample representing only an *amount* of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray. The contrast ranges from black at the weakest intensity to white at the strongest.

- We use the linear equation to get the intensity of the image from RGB channels, Like
  $$
  {\displaystyle Y_{\mathrm {linear} }=0.07R_{\mathrm {linear} }+0.7152G_{\mathrm {linear} }+0.21B_{\mathrm {linear} }}.
  $$

<img src="Images\grayscale_image.png" style="zoom:80%;" />

#### Histogram of R,G,B channels

- An **image histogram** is a type of histogram that acts as graphical representation of the tonal distribution in digital image. It plots the number of pixels for each tonal value.

- The horizontal axis of the graph represents the tonal variations, while the vertical axis represents the total number of pixels in that particular tone.

  - Histogram of **Red** channel in image:

    <img src="Images\red_channel_histogram.png" style="zoom:100%;" />

  - Histogram of **Green** channel:

    <img src="Images\green_channel_image.png" style="zoom:100%;" />

  - Histogram of **Blue** channel:

    <img src="Images\blue_channel_histogram.png" style="zoom:100%;" />

#### Cumulative Curve

- To draw the **cumulative curve**, we have to compute the cumulative sum of the histogram. The cumulative sum is exactly as it sounds — the sum of all values in the histogram up to that point, taking into account all previous values.

- Then, We now have the cumulative sum, but as you can see, the values are huge (> 6,000,000). We’re going to be matching these values to our original image in the final step, so we have to normalize them to conform to a range of 0–255.

  <img src="E:\Computer vision tasks\Renad part\report\Images\cumulative_curve.png" style="zoom:100%;" />

## Hybrid Image

- A **hybrid image** is an image that is perceived in one of two different ways, depending on viewing distance, based on the way humans process visual input.

- In this technique, you have to have two images in the same size and shape, one is applied with **Gaussian** Filter and another is applied with **Laplacian** filter. Then, use filtered images to get new image by using this equation:
  $$
  H = I_1 · \alpha  + I_2 ·(1 − \alpha)
  $$

  - First image: 

    <img src="Images\test.jpg" style="zoom:50%;" />

  - Second image:

    <img src="Images\test2.jpg" style="zoom:50%;" />

  - **Hybrid** image: 

    <img src="Images\hybrid_image.png" style="zoom:70%;" />