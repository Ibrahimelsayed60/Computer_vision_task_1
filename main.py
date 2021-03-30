from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from task1 import Ui_MainWindow
import sys
import random
import cv2
import numpy as np
import pyqtgraph as pg

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        path = 'lena.jpg'
        self.img = cv2.imread(path, 0)
        self.img2 = cv2.rotate(cv2.imread(path),cv2.ROTATE_90_CLOCKWISE)
        self.img3 = cv2.rotate(cv2.imread(path,0),cv2.ROTATE_90_CLOCKWISE)
        self.img4 = cv2.rotate(cv2.imread('test.jpg',0),cv2.ROTATE_90_CLOCKWISE)
        self.img5 = cv2.rotate(cv2.imread('test2.jpg',0),cv2.ROTATE_90_CLOCKWISE)
        ###########To remove axis###############
        self.ui.widget_2.getPlotItem().hideAxis('bottom')
        self.ui.widget_2.getPlotItem().hideAxis('left')
        self.ui.widget_3.getPlotItem().hideAxis('bottom')
        self.ui.widget_3.getPlotItem().hideAxis('left')

        self.ui.image1.setPixmap(QPixmap(path))
        self.ui.comboBox_Image1.currentIndexChanged[int].connect(self.noisy_image)
        self.ui.comboBox_Image1_3.currentIndexChanged[int].connect(self.filtered_image)
        self.ui.comboBox_Image1_2.currentIndexChanged[int].connect(self.threshold_image)
        self.Hybrid_image(self.img4,self.img5,0.3)

    def convolution(self, image, kernel, average=False, verbose=False):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            None
        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape	 
        output = np.zeros(image.shape)	 
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)	 
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))	 
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image	 
        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]
        return output

    # Add additive noise to the image
    def gaussian_noise(self, img):
        mean = 0
        var = 10
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (256, 256))
        new_img = np.zeros(img.shape, np.float32)
        if len(img.shape) == 2:
            new_img = img + gaussian
        else:
            new_img[:, :, 0] = img[:, :, 0] + gaussian
            new_img[:, :, 1] = img[:, :, 1] + gaussian
            new_img[:, :, 2] = img[:, :, 2] + gaussian
        cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        return new_img

    def salt_and_pepper_noise(self, img):
        row , col = img.shape
        # Randomly pick some pixels in the image for coloring them white
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):	        
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)	          
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)	          
            # Color that pixel to white
            img[y_coord][x_coord] = 255	          
        # Randomly pick some pixels in the image for coloring them black
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):	        
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)	          
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)	          
            # Color that pixel to black
            img[y_coord][x_coord] = 0	          
        return img
    
    # Filter the noisy image using the low pass filters

    def averaging_filter(self, img):
        row, col = img.shape
        # Develop Averaging filter(3, 3) mask
        mask = np.ones([3, 3], dtype = int)
        mask = mask / 9
        # Convolve the 3X3 mask over the image
        img_new = self.convolution(img, mask)
        return img_new

    def gaussian_filter(self, img):
        sigma = 0.5
        row, col = img.shape
        filter_size = 2 * int(4 * sigma + 0.5) + 1
        gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
        m = filter_size//2
        n = filter_size//2
        for x in range(-m, m+1):
            for y in range(-n, n+1):
                x1 = 2*np.pi*(sigma**2)
                x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
                gaussian_filter[x+m, y+n] = x2/x1
        img_new = self.convolution(img, gaussian_filter)
        return img_new

    def median_filter(self, img):
        # Traverse the image. For every 3X3 area,find the median of the pixels and replace the center pixel by the median
        row, col = img.shape
        img_new = np.zeros([row, col])
        for i in range(1, row-1):
            for j in range(1, col-1):
                temp = [img[i-1, j-1], img[i-1, j], img[i-1, j + 1], img[i, j-1], img[i, j], img[i, j + 1], img[i + 1, j-1], img[i + 1, j], img[i + 1, j + 1]]
                temp = sorted(temp)
                img_new[i, j]= temp[4]
        return img_new

    def noisy_image(self):
        if self.ui.comboBox_Image1.currentIndex() == 1:
            self.image = self.gaussian_noise(self.img)
        elif self.ui.comboBox_Image1.currentIndex() == 2:
            self.image = self.salt_and_pepper_noise(self.img)
        self.ui.image2.setPixmap(QPixmap(self.display(self.image)))
        self.filtered_image()
        

    def filtered_image(self):
        if self.ui.comboBox_Image1_3.currentIndex() == 0:
            output = self.averaging_filter(self.image)
        elif self.ui.comboBox_Image1_3.currentIndex() == 1:
            output = self.gaussian_filter(self.image)
        elif self.ui.comboBox_Image1_3.currentIndex() == 2:
            output = self.median_filter(self.image)
        self.ui.image3.setPixmap(QPixmap(self.display(output)))

    def display(self, img):
            img = np.array(img).reshape(self.img.shape[1],self.img.shape[0]).astype(np.uint8)
            img = QtGui.QImage(img, img.shape[0],img.shape[1],QtGui.QImage.Format_Grayscale8)
            return img

########################Threshold################################
    def global_threshold_v_127(self,img):
        thresh = 100
        binary = img > thresh
        for i in range(0,len(binary),1):
            for j in range(0,len(binary),1):
                if binary[i][j] == True:
                    binary[i][j] = 256
                else:
                    binary[i][j]=0

        return binary

    def local_treshold(self,input_img):
        h, w = input_img.shape
        S = w/8
        s2 = S/2
        T = 15.0
        #integral img
        int_img = np.zeros_like(input_img, dtype=np.uint32)
        for col in range(w):
            for row in range(h):
                int_img[row,col] = input_img[0:row,0:col].sum()
        #output img
        out_img = np.zeros_like(input_img)    
        for col in range(w):
            for row in range(h):
                #SxS region
                y0 = int(max(row-s2, 0))
                y1 = int(min(row+s2, h-1))
                x0 = int(max(col-s2, 0))
                x1 = int(min(col+s2, w-1))
                count = (y1-y0)*(x1-x0)
                sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]
                if input_img[row, col]*count < sum_*(100.-T)/100.:
                    out_img[row,col] = 0
                else:
                    out_img[row,col] = 255
        return out_img

    def Transormation_to_grayScale(self,input_image):
        H,W = input_image.shape[:2]
        gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.07 * input_image[i,j,0]  + 0.72 * input_image[i,j,1] + 0.21 * input_image[i,j,2], 0, 255)
        return gray

    def threshold_image(self):
        if self.ui.comboBox_Image1_2.currentIndex() == 1:
            out_put = (self.global_threshold_v_127(self.img3))
            my_img = pg.ImageItem(out_put)
            self.ui.widget_2.addItem(my_img)
        elif self.ui.comboBox_Image1_2.currentIndex() == 2:
            out = self.local_treshold(self.img3)
            my_img = pg.ImageItem(out)
            self.ui.widget_2.addItem(my_img)
        elif self.ui.comboBox_Image1_2.currentIndex() == 3:
            out = self.Transormation_to_grayScale(self.img2)
            my_img = pg.ImageItem(out)
            self.ui.widget_2.addItem(my_img)
        else:
            pass


    
    #######################Histograms#################################
    def cumulative_histogram(self,input_color_image):
        pass

    ############################Hybrid Image###########################  
    def laplacian_image(self,input_image):
        input_image -= np.amin(input_image) #map values to the (0, 255) range
        A =input_image* 255.0/np.amax(input_image)

        #Kernel for negative Laplacian
        kernel = np.ones((3,3))*(-1)
        kernel[1,1] = 8

        #Convolution of the image with the kernel:
        Lap = self.convolution(A, kernel)

        return Lap


    def Hybrid_image(self,first_image,second_image,alpha):
        if first_image.shape[0] == second_image.shape[0] and first_image.shape[1] == second_image.shape[1]:
            laplace_image = self.laplacian_image(first_image)
            gaussian_image = self.gaussian_filter(second_image)
            Hybrid_img = alpha * laplace_image + (1 - alpha) * gaussian_image
            # img = np.array(Hybrid_img).reshape(self.img4.shape[1],self.img4.shape[0]).astype(np.uint8)
            # img = QtGui.QImage(img, img.shape[0],img.shape[1],QtGui.QImage.Format_Grayscale8)
            my_img = pg.ImageItem(Hybrid_img)
            self.ui.widget_3.addItem(my_img)

        else:
            pass




def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()