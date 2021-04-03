   #######################Edge Detection#################################
    def Transormation_to_grayScale(self,input_image):
        H,W = input_image.shape[:2]
        self.gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                self.gray[i,j] = np.clip(0.07 * input_image[i,j,0]  + 0.72 * input_image[i,j,1] + 0.21 * input_image[i,j,2], 0, 255)
        return self.gray

    def edge_detection(self):
        if self.ui.comboBox_3.currentIndex() == 1:
            output = self.sobel(self.img)
        elif self.ui.comboBox_3.currentIndex() == 2:
            output = self.prewitt(self.img)
        elif self.ui.comboBox_3.currentIndex() == 3:
            output = self.roberts(self.img)
        self.ui.widget_4.setPixmap(QPixmap(self.display2(output)))

    def display2(self, img):
            img = np.array(img).reshape(self.img.shape[1],self.img.shape[0]).astype(np.uint8)
            img = QtGui.QImage(img, img.shape[0] ,img.shape[1] ,QtGui.QImage.Format_Grayscale8)
            return img


    def sobel(self,img):
        [nx, ny, nz] = np.shape(self.img)  # nx: height, ny: width, nz: colors (RGB)

        # Extracting each one of the RGB components
        r_img, g_img, b_img = self.img[:, :, 0],self.img[:, :, 1], self.img[:, :, 2]

        # The following operation will take weights and parameters to convert the color image to grayscale
        gamma = 1.400  # a parameter
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
        grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma


        # Here we define the matrices associated with the Sobel filter
        Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        [rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
        sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
                gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
        return sobel_filtered_image