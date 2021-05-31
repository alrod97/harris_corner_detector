## Harris Corner Detector Implementation from scratch
The main goal is to implement the Harris corner detector from scratch using finite differences matrixes, convolution operations, reshaping arrays or solving linear system of equations.

The main approach is as follows:
1) Compute image gradients I_x and I_y using finite differences matrixes. We use central differences where possible and forward/backward differences in the image boundaries. This should only affect our solution if we want to trace boundary-near points.
2) Compute each element of the structure tensor M. M_1_1 as a 2D field. In other words: M_1_1 at all pixel points. This is done by simply multiplying both 2D matrixes M_1_1 = I_x * I_x. We do the same for M_2_2 and M_1_2=M_2_1
3) We convolve each field M_i_j with a Gaussian kernel which width we need to specify
4) Next, for every pixel point we compute the scoring function: C = det(M) - k*trace(M)^2 which results in a 2D Matrix where we need to specify k
5) Finally, we set a trheshold to filter the corners with θ only consider as corner points those with C>θ 

** A more detailed file containing the explanations on how we compute everything will follow, specially the finite difference matrixes  :)

## Example

We consider the next image as an example:
![Original](/results_images/Original_grayscale_image.png)

The gradient I_x result:
![I_x](/results_images/x-gradient.png)

The gradient I_y result:
![I_y](/results_images/y-gradient.png)

The C values after applying the threshold:
![C](/results_images/C_after_threshold.png)

The computed Image corners (we have to look closer to see the black points corners :), will fix it for a better visualisation) :
![corners](/results_images/Image_with_corners.png)
