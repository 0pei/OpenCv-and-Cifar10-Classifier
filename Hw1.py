import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.img4=cv2.imread('.\Q4_Image\SQUARE-01.png')
        # self.ui.teImageSize.setText("("+str(self.img4.shape[0])+","+str(self.img4.shape[1])+")")
        # self.ui.teTranslation.setText("(0,0)")
        # self.ui.teAngle.setText("10")
        # self.ui.teScale.setText("0.5")
        # self.ui.teWindowSize.setText("(400,300)")
        self.ui.btnLoadImage.clicked.connect(self.btnLoadImageClicked)
        self.ui.btnColorSeparation.clicked.connect(self.btnColorSeparationClicked)
        self.ui.btnColorTransformation.clicked.connect(self.btnColorTransformationClicked)
        self.ui.btnBlending.clicked.connect(self.btnBlendingClicked)
        self.ui.btnGaussianBlur.clicked.connect(self.btnGaussianBlurClicked)
        self.ui.btnBilateralFilter.clicked.connect(self.btnBilateralFilterClicked)
        self.ui.btnMedianFilter.clicked.connect(self.btnMedianFilterClicked)
        self.ui.btnGaussianBlur2.clicked.connect(self.btnGaussianBlur2Clicked)
        self.ui.btnSobelX.clicked.connect(self.btnSobelXClicked)
        self.ui.btnSobelY.clicked.connect(self.btnSobelYClicked)
        self.ui.btnMagnitude.clicked.connect(self.btnMagnitudeClicked)
        self.ui.btnResize.clicked.connect(self.btnResizeClicked)
        self.ui.btnTranslation.clicked.connect(self.btnTranslationClicked)
        self.ui.btnRotationScaling.clicked.connect(self.btnRotationScalingClicked)
        self.ui.btnShearing.clicked.connect(self.btnShearingClicked)
    # 1.1
    def btnLoadImageClicked(self):
        img1=cv2.imread('.\Q1_Image\Sun.jpg')
        cv2.imshow('Hw1-1', img1)
        print("Height : ", img1.shape[0], "\nWidth : ", img1.shape[1])
    # 1.2
    def btnColorSeparationClicked(self):
        img1=cv2.imread('.\Q1_Image\sun.jpg')
        B, G, R=cv2.split(img1)
        zeros=np.zeros(img1.shape[:2],dtype=img1.dtype) # uint8
        cv2.imshow('B channel', cv2.merge([B,zeros,zeros]))
        cv2.imshow('G channel', cv2.merge([zeros,G,zeros]))
        cv2.imshow('R channel', cv2.merge([zeros,zeros,R]))
        # cv2.imshow('Merge test', cv2.merge([B,G,R]))        # original img
    # 1.3
    def btnColorTransformationClicked(self):
        img1=cv2.imread('.\Q1_Image\Sun.jpg')
        B, G, R=cv2.split(img1)
        cv2.imshow('I1', cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        cv2.imshow('I2', np.uint8(R/3+G/3+B/3))             # upper limit 256
        # rows,cols,_ = img1_1.shape
        # zeros = np.zeros((rows,cols),dtype=img1_1.dtype)
        # for y in range(rows):
        #     for x in range(cols):
        #         avg = sum(img1_1[y,x])/3
        #         zeros[y,x] = np.uint8(avg)
        # cv2.imshow('I2', zeros)
    # 1.4
    def btnBlendingClicked(self):
        top=cv2.imread('.\Q1_Image\Dog_Strong.jpg')
        bottom=cv2.imread('.\Q1_Image\Dog_Weak.jpg')
        cv2.namedWindow('Blend')
        cv2.imshow('Blend', top)
        def update(x):
            weight=x/255
            cv2.imshow('Blend', cv2.addWeighted(top, 1-weight, bottom, weight, 0))
        cv2.createTrackbar('Blend', 'Blend', 0, 255, update)
        cv2.setTrackbarPos('Blend', 'Blend', 127)
    # 2.1
    def btnGaussianBlurClicked(self):
        img2=cv2.imread('.\Q2_Image\Lenna_whiteNoise.jpg')
        cv2.imshow('Origin image', img2)
        cv2.imshow('Gaussian Blur', cv2.GaussianBlur(img2, (5, 5), 0))          # 5x5
    # 2.2
    def btnBilateralFilterClicked(self):
        img2=cv2.imread('.\Q2_Image\Lenna_whiteNoise.jpg')
        cv2.imshow('Origin image', img2)
        cv2.imshow('Bilateral Filter', cv2.bilateralFilter(img2, 9, 90, 90))   # sigmaColor, sigmaSpaces
    # 2.3
    def btnMedianFilterClicked(self):
        img2=cv2.imread('.\Q2_Image\Lenna_pepperSalt.jpg')
        cv2.imshow('Origin image', img2)
        cv2.imshow('Median Filter 3x3', cv2.medianBlur(img2, 3))
        cv2.imshow('Median Filter 5x5', cv2.medianBlur(img2, 5))
    # 3.1
    def btnGaussianBlur2Clicked(self):
        img3=cv2.imread('.\Q3_Image\House.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imshow('House.jpg', cv2.imread('.\Q3_Image\House.jpg'))
        cv2.imshow('Grayscale', img3)
        column, row = img3.shape[0], img3.shape[1]
        self.GaussianBlur=np.array([range(row) for _ in range(column)])
        y, x=np.mgrid[-1:2, -1:2]
        gaussian_kernel=np.exp(-(x**2+y**2))
        Gnorm=gaussian_kernel/gaussian_kernel.sum()
        for height in range(img3.shape[1]):
            for width in range(img3.shape[0]):
                if width==0 or height==0 or width==img3.shape[0]-1 or height==img3.shape[1]-1:
                    self.GaussianBlur[width][height]=img3[width][height]
                else:
                    # conv=[[img3[width-1][height-1],img3[width-1][height],img3[width-1][height+1]],
                    #       [img3[width][height-1],img3[width][height],img3[width][height+1]],
                    #       [img3[width+1][height-1],img3[width+1][height],img3[width+1][height+1]]]
                    #  conv=[img3[width-1][height-1:height+2],img3[width][height-1:height+2],img3[width+1][height-1:height+2]]
                     # self.GaussianBlur[width][height]=(conv*Gnorm).sum()
                    self.GaussianBlur[width][height]=(img3[width-1:width+2,height-1:height+2]*Gnorm).sum()
        self.GaussianBlur=self.GaussianBlur.astype(np.uint8)
        cv2.imshow('Gaussian Blur', self.GaussianBlur)
    # 3.2
    def btnSobelXClicked(self):
        column, row = self.GaussianBlur.shape[0], self.GaussianBlur.shape[1]
        self.SobelX=np.array([range(row) for _ in range(column)])
        SobelXFilter=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        for height in range(self.GaussianBlur.shape[1]):
            for width in range(self.GaussianBlur.shape[0]):
                if width==0 or height==0 or width==self.GaussianBlur.shape[0]-1 or height==self.GaussianBlur.shape[1]-1:
                    self.SobelX[width][height]=self.GaussianBlur[width][height]
                else:
                    self.SobelX[width][height]=abs((self.GaussianBlur[width-1:width+2,height-1:height+2]*SobelXFilter).sum())
        self.SobelX=np.clip(self.SobelX,0,255)
        self.SobelX=self.SobelX.astype(np.uint8)
        cv2.imshow('Sobel X', self.SobelX)
    # 3.3    
    def btnSobelYClicked(self):
        column, row = self.GaussianBlur.shape[0], self.GaussianBlur.shape[1]
        self.SobelY=np.array([range(row) for _ in range(column)])
        SobelYFilter=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        for height in range(self.GaussianBlur.shape[1]):
            for width in range(self.GaussianBlur.shape[0]):
                if width==0 or height==0 or width==self.GaussianBlur.shape[0]-1 or height==self.GaussianBlur.shape[1]-1:
                    self.SobelY[width][height]=self.GaussianBlur[width][height]
                else:
                    self.SobelY[width][height]=abs((self.GaussianBlur[width-1:width+2,height-1:height+2]*SobelYFilter).sum())
        self.SobelY=np.clip(self.SobelY,0,255)
        self.SobelY=self.SobelY.astype(np.uint8)
        cv2.imshow('Sobel Y', self.SobelY)
    # 3.4   
    def btnMagnitudeClicked(self):
        column, row = self.GaussianBlur.shape[0], self.GaussianBlur.shape[1]
        Magnitude=np.array([range(row) for _ in range(column)])
        for height in range(self.GaussianBlur.shape[1]):
            for width in range(self.GaussianBlur.shape[0]):
                Magnitude[width][height]=math.sqrt(math.pow(self.SobelX[width][height],2)+math.pow(self.SobelY[width][height],2))
        Magnitude=Magnitude/Magnitude.max()
        cv2.imshow('Magnitude', Magnitude)
    # 4.1
    def btnResizeClicked(self):
        # size=eval(self.ui.teImageSize.toPlainText())            # string to tuple
        self.img4=cv2.resize(self.img4, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imshow('img_1', self.img4)
        # cv2.imshow('NEAREST', cv2.resize(self.img4, size, interpolation=cv2.INTER_NEAREST))
        # cv2.imshow('LINEAR', cv2.resize(self.img4, size, interpolation=cv2.INTER_LINEAR))
        # cv2.imshow('AREA', cv2.resize(self.img4, size, interpolation=cv2.INTER_AREA))
        # cv2.imshow('CUBIC', cv2.resize(self.img4, size, interpolation=cv2.INTER_CUBIC))
        # cv2.imshow('LANCZOS4', cv2.resize(self.img4, size, interpolation=cv2.INTER_LANCZOS4))
    # 4.2 
    def btnTranslationClicked(self):
        # translation=eval(self.ui.teTranslation.toPlainText())
        # windowSize=eval(self.ui.teWindowSize.toPlainText())
        M1 = np.float32([[1, 0, 0], [0, 1, 60]])
        self.img4=cv2.warpAffine(self.img4, M1, (400, 300))
        cv2.imshow('img_2', self.img4)
    # 4.3
    def btnRotationScalingClicked(self):
        center=(self.img4.shape[0]/2, self.img4.shape[1]/2)     # (width/2, height/2)
        # angle=int(self.ui.teAngle.toPlainText())
        # scale=float(self.ui.teScale.toPlainText())
        M2 = cv2.getRotationMatrix2D(center, 10, 0.5)           # center, angle, scale
        self.img4=cv2.warpAffine(self.img4, M2, (400, 300))
        cv2.imshow('img_3', self.img4)
    # 4.4
    def btnShearingClicked(self):
        OldLocation = np.float32([[50,50],[200,50],[50,200]])
        NewLocation = np.float32([[10,100],[200,50],[100,250]])
        M3=cv2.getAffineTransform(OldLocation, NewLocation)     # 仿射變換
        self.img4=cv2.warpAffine(self.img4, M3, (400, 300))
        cv2.imshow('img_4', self.img4)
        
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())