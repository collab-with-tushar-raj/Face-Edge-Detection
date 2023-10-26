# Face Edge Detection

This work describes various edge-detection approaches for my passport image. **My aim is to
delineate only the face from the image.** The work has been compared with **RGB thresholding,
HSV thresholding, K-means clustering and SVM**.
The image is an **RGB image** having a **red intensity value of up to 223** , a **green intensity value of
up to 160** and a **blue intensity value of up to 130**. I first applied thresholding by considering image
indexes greater than 130 to the RGB image which gives the better visibility of my face. I then
segmented my face by removing the various noises. I then applied edge detection to different
methods.
The **hue, saturation and value thresholds are 0.9, 0.5 and 0.07 respectively.** I first did the image
processing by using these thresholds and then segmented my face to show a mask image. I removed
noise below the chin with the help of **improfile.** I then applied edge detection of my segmented face
to different methods.
Applied **clustering for 2, 3, 5, 7 and 10 regions in the K-Means approach**. I then used class 1 for
my face to show the HSV and RGB images. I finally segmented my face from the image by removing
the various noises and then applied edge detection to different methods.
Applied **the SVM approach and predicted the HSV and RGB image of my segmented face**. I
segmented the face by removing the various noises and then applied SVM algorithm. The loss for
predicting the HSV and RGB images is nearly 0. I then applied edge detection to the segmented
predicted face to different methods.
After applying all the above approaches and different edge detection methods like **Sobel, Roberts,
Canny, Log, Zero Cross, Prewitt and my own edge method** , I have found that SVM has
outperformed the others. In RGB and HSV thresholding, we see a lot of noise in every edge method
and even the breaking of the face edge in Sobel, Roberts, and Prewitt methods (for RGB
thresholding). The K-Means algorithm has failed to distinguish between my face and the background
(even for k=10 regions). The Sobel, Roberts and Prewitt methods have again shown the breaking of
the face edge, and the other methods show huge noises. In the SVM approach, we see less noise in
Sobel, Roberts, and Prewitt's methods but there are breaking face edges. We see many noises in the
Canny, Log and Zero Cross methods. But in my own method, the noise and breaking face edges are
comparatively less. Even the eyes and lips can be seen clearly. Therefore, my method in SVM has
outperformed all the existing methods. I then removed the noises inside my face like eyes, nose, and
lips. The result of my method looks like below –

![image](https://github.com/collab-with-tushar-raj/Face-Edge-Detection/assets/39027684/9b4b3c54-e8e9-4c37-bb1b-83307ed5d6fd)



In conclusion, the challenge I have found in detecting my facial edge is mainly to keep the continuity
of facial edge lines and not let them break anywhere. Many close noises to the face like ears, hairs
and shoulders have been removed with the help of **improfile**. My own method with the SVM
approach has performed well in comparison with the other approaches but could have been more
fine-tuned to get the perfect facial edge with the accurate convolution operator. Also, perfectly
removing noises within the face was a challenging task.

## Extra Features Added: -

- Added my own signature to my image.
 ![image](https://github.com/collab-with-tushar-raj/Face-Edge-Detection/assets/39027684/0c0f15f9-80b9-4ec8-b408-9b704d641684)
- Created nested functions and clear interfaces.
- Coloured the detected face edge in all the different approaches and edge methods.
- The entire work is in a single **.m** file.
- All the images have been displayed in 1 figure whose name is **Face Edge Detection**.
- Created my own edge method that stood out the best among all the existing methods in SVM
    approach.


