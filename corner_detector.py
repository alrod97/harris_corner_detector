import numpy as np
import scipy as sp
from scipy.sparse import diags
import matplotlib.pyplot as plt
import scipy.signal as sg
import cv2

class HarrisCornerDetector():
    def __init__(self, sigma=1, k=0.05, threshold=10e-7):
        self.sigma = sigma
        self.k = k
        self.threshold = threshold

        self.I_x = None
        self.I_y = None
        self.img = None
        self.img_with_corners = None

    def get_x_gradient(self, img):
        # read dimensions
        Ny, Nx = img.shape

        # Create sparse D_x matrix
        D_x = self.D_x_matrix(Nx, Ny)

        # Get I_x gradient
        I_x_vector = D_x @ np.reshape(img, -1)

        I_x = np.reshape(I_x_vector, (Ny, Nx))
        return I_x

    def get_y_gradient(self, img):
        # read dimensions
        Ny, Nx = img.shape

        # Create sparse D_x matrix
        D_y = self.D_y_matrix(Nx, Ny)

        # Get I_x gradient
        I_y_vector = D_y @ np.reshape(img.T, -1)

        I_y_t = np.reshape(I_y_vector, (Nx, Ny))
        I_y = I_y_t.T
        return I_y

    def D_x_matrix(self, n_x, n_y):
        v = np.ones(n_x - 1)
        diagonals = [-v, v]
        d_x_small = diags(diagonals, [-1, 1]).toarray()
        d_x_small[0, 0] = -2
        d_x_small[0, 1] = 2
        d_x_small[-1, -1] = 2
        d_x_small[-1, n_x - 2] = -2

        D_x = sp.sparse.kron(sp.sparse.eye(n_y), d_x_small)

        return 1 / 2 * D_x

    def D_y_matrix(self, n_x, n_y):
        v = np.ones(n_y - 1)
        diagonals = [v, -v]
        d_y_small = diags(diagonals, [-1, 1]).toarray()
        d_y_small[0, 0] = 2
        d_y_small[0, 1] = -2
        d_y_small[-1, -1] = -2
        d_y_small[-1, n_y - 2] = 2

        D_y = sp.sparse.kron(sp.sparse.eye(n_x), d_y_small)

        return 1 / 2 * D_y

    def generate_gaussian_kernel(self, sigma=1, mu=0):
        k = int(2 * 2 * sigma + 1)
        x, y = np.meshgrid(np.linspace(-1, 1, k), np.linspace(-1, 1, k))
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        return g

    def non_max_surpression(self, A):
        # get elements non zero candidates
        positions = np.argwhere(A != 0)
        A_surpressed = np.zeros_like(A)
        rows, cols = A.shape

        for pos in positions:
            cur_value = A[pos[0], pos[1]]
            if cur_value == 0:
                pass
            else:
                # do not check boundaries == set to zero
                if pos[0] == (rows - 1) or pos[0] == 0:
                    pass
                elif pos[1] == (cols - 1) or pos[1] == 0:
                    pass
                else:
                    if cur_value < A[pos[0] + 1, pos[1]]:
                        pass
                    elif cur_value < A[pos[0] - 1, pos[1]]:
                        pass
                    elif cur_value < A[pos[0], pos[1] + 1]:
                        pass
                    elif cur_value < A[pos[0], pos[1] - 1]:
                        pass
                    else:
                        A_surpressed[pos[0], pos[1]] = cur_value
        return A_surpressed


    def apply(self, image_path):
        # load  image as type np array grayscale (read only first channel)
        self.img = cv2.imread(image_path, 0)

        # normalize image
        img_normalized = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        # get image gradients
        self.I_x = self.get_x_gradient(img_normalized)
        self.I_y = self.get_y_gradient(img_normalized)

        # get gaussian filter
        gaussian = self.generate_gaussian_kernel(sigma=self.sigma)

        # Compute I_x^2, I_y^2 and I_xy
        I_x2 = self.I_x * self.I_x
        I_y2 = self.I_y * self.I_y
        I_xy = self.I_x * self.I_y

        # convolve with same padding gaussian with I_x2, I_y2 and I_xy to get entries of every tensor
        M_1_1_entries = sg.convolve2d(I_x2, gaussian, 'same', boundary='symm')
        M_2_2_entries = sg.convolve2d(I_y2, gaussian, 'same', boundary='symm')
        M_1_2_entries = sg.convolve2d(I_xy, gaussian, 'same', boundary='symm')

        # Reshape vectors to 1d
        M_1_1_entries_1d = np.reshape(M_1_1_entries, -1)
        M_2_2_entries_1d = np.reshape(M_2_2_entries, -1)
        M_1_2_entries_1d = np.reshape(M_1_2_entries, -1)

        # Init C array to store values of determinant
        C_1d = np.zeros_like(M_1_1_entries_1d)

        for pixel in range(int(self.I_x.size)):
            M_matrix = np.zeros((2, 2))
            M_matrix[0, 0] = M_1_1_entries_1d[pixel]
            M_matrix[1, 1] = M_2_2_entries_1d[pixel]
            M_matrix[0, 1] = M_1_2_entries_1d[pixel]
            M_matrix[1, 0] = M_1_2_entries_1d[pixel]
            C_1d[pixel] = np.linalg.det(M_matrix) - self.k * np.trace(M_matrix) ** 2

        # reshape C matrix to 2d
        self.C = np.reshape(C_1d, self.I_x.shape)

        # Apply threshold
        C_thres = self.C.copy()
        C_thres[self.C < self.threshold] = 0

        # Non max surpression

        self.C_surspresed = self.non_max_surpression(C_thres)
        self.C_surspresed[self.C_surspresed != 0] = 255

        # Show corners on original image
        self.img_with_corners = self.img.copy()
        self.img_with_corners[self.C_surspresed != 0] = 0

    def plot_results(self, original_image = True, gradient_x=True, gradient_y= True, C_before_threshold=True, C_after_threshold_surpressed=True):
        plt.figure('Image with corners')
        plt.imshow(self.img_with_corners, cmap='gray')
        plt.colorbar()

        if original_image:
            plt.figure('Original grayscale image')
            plt.imshow(self.img, cmap='gray')
            plt.colorbar()

        if gradient_x:
            plt.figure('x-gradient')
            plt.imshow(self.I_x)
            plt.colorbar()

        if gradient_y:
            plt.figure('y-gradient')
            plt.imshow(self.I_y)
            plt.colorbar()

        if C_before_threshold:
            plt.figure('C')
            plt.imshow(self.C)
            plt.colorbar()

        if C_after_threshold_surpressed:
            plt.figure('C _final')
            plt.imshow(self.C_surspresed)
            plt.colorbar()

        plt.show()

if __name__ == "__main__":
    example = HarrisCornerDetector(threshold=10e-5)
    example.apply(image_path='images/img1.png')
    example.plot_results()