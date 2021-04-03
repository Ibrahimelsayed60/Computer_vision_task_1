
Renad Taher <renad.taher12@gmail.com>
15:09 (6 hours ago)
to me

if filterType=="LPF":
            self.nrow, self.ncol = data.shape
            mask = np.zeros((self.nrow,self.ncol))
            avgMask= np.ones((9,9))/81
            mask[24:33,24:33] = avgMask
            mask = np.fft.fftshift(np.fft.fft2(mask))
            img = np.fft.fftshift(np.fft.fft2(data))
            newImg = np.fft.ifft2(mask*img)
            newImg = np.log(1+np.abs(newImg))
            return(newImg)
        
        if filterType=="HPF":
            self.nrow, self.ncol = data.shape
            mask = np.zeros((self.nrow,self.ncol))
            hpf = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
            mask[20:23,20:23] = hpf
            mask = np.fft.fftshift(np.fft.fft2(mask))
            img = np.fft.fftshift(np.fft.fft2(data))
            newImg = np.fft.ifft2(mask*img)
            newImg = np.log(1+np.abs(newImg))
            return(newImg)
