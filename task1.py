# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'task1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(969, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame_16 = QtWidgets.QFrame(self.centralwidget)
        self.frame_16.setGeometry(QtCore.QRect(10, 10, 921, 551))
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_16)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.splitter = QtWidgets.QSplitter(self.frame_16)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.frame_7 = QtWidgets.QFrame(self.splitter)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_7)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.frame_3 = QtWidgets.QFrame(self.frame_7)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.frame_3)
        self.frame_2.setMaximumSize(QtCore.QSize(16777215, 45))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setFamily("Serif")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.comboBox_Image1 = QtWidgets.QComboBox(self.frame_2)
        self.comboBox_Image1.setObjectName("comboBox_Image1")
        self.comboBox_Image1.addItem("")
        self.comboBox_Image1.addItem("")
        self.comboBox_Image1.addItem("")
        self.comboBox_Image1.addItem("")
        self.comboBox_Image1.setItemText(3, "")
        self.horizontalLayout_2.addWidget(self.comboBox_Image1)
        self.comboBox_Image1_3 = QtWidgets.QComboBox(self.frame_2)
        self.comboBox_Image1_3.setObjectName("comboBox_Image1_3")
        self.comboBox_Image1_3.addItem("")
        self.comboBox_Image1_3.addItem("")
        self.comboBox_Image1_3.addItem("")
        self.comboBox_Image1_3.addItem("")
        self.comboBox_Image1_3.setItemText(3, "")
        self.horizontalLayout_2.addWidget(self.comboBox_Image1_3)
        self.comboBox_Image1_2 = QtWidgets.QComboBox(self.frame_2)
        self.comboBox_Image1_2.setObjectName("comboBox_Image1_2")
        self.comboBox_Image1_2.addItem("")
        self.comboBox_Image1_2.addItem("")
        self.comboBox_Image1_2.addItem("")
        self.comboBox_Image1_2.addItem("")
        self.comboBox_Image1_2.setItemText(3, "")
        self.horizontalLayout_2.addWidget(self.comboBox_Image1_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.frame_2)
        self.frame = QtWidgets.QFrame(self.frame_3)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.image2 = QtWidgets.QLabel(self.frame)
        self.image2.setText("")
        self.image2.setObjectName("image2")
        self.gridLayout.addWidget(self.image2, 0, 1, 1, 1)
        self.image1 = QtWidgets.QLabel(self.frame)
        self.image1.setText("")
        self.image1.setObjectName("image1")
        self.gridLayout.addWidget(self.image1, 0, 0, 1, 1)
        self.image3 = QtWidgets.QLabel(self.frame)
        self.image3.setText("")
        self.image3.setObjectName("image3")
        self.gridLayout.addWidget(self.image3, 0, 2, 1, 1)
        self.image4 = QtWidgets.QLabel(self.frame)
        self.image4.setText("")
        self.image4.setObjectName("image4")
        self.gridLayout.addWidget(self.image4, 0, 3, 1, 1)
        self.verticalLayout_3.addWidget(self.frame)
        self.gridLayout_3.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_7.addWidget(self.frame_3)
        self.frame_4 = QtWidgets.QFrame(self.frame_7)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_5 = QtWidgets.QFrame(self.frame_4)
        self.frame_5.setMaximumSize(QtCore.QSize(16777215, 45))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setFamily("Serif")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.label_2 = QtWidgets.QLabel(self.frame_5)
        font = QtGui.QFont()
        font.setFamily("Serif")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.gridLayout_5.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.frame_5)
        self.frame_6 = QtWidgets.QFrame(self.frame_4)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_13 = QtWidgets.QFrame(self.frame_6)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.frame_14 = QtWidgets.QFrame(self.frame_13)
        self.frame_14.setGeometry(QtCore.QRect(0, -1, 411, 151))
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.frame_15 = QtWidgets.QFrame(self.frame_13)
        self.frame_15.setGeometry(QtCore.QRect(410, 0, 421, 151))
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.verticalLayout_6.addWidget(self.frame_13)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.gridLayout_6.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.frame_6)
        self.gridLayout_4.addLayout(self.verticalLayout_4, 0, 0, 1, 1)
        self.verticalLayout_7.addWidget(self.frame_4)
        self.gridLayout_7.addLayout(self.verticalLayout_7, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 969, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "Original Image"))
        self.comboBox_Image1.setItemText(0, _translate("MainWindow", "Add Noise"))
        self.comboBox_Image1.setItemText(1, _translate("MainWindow", "Gaussian Noise"))
        self.comboBox_Image1.setItemText(2, _translate("MainWindow", "Salt and Pepper Noise"))
        self.comboBox_Image1_3.setItemText(0, _translate("MainWindow", "Average Filter"))
        self.comboBox_Image1_3.setItemText(1, _translate("MainWindow", "Gaussian Filter"))
        self.comboBox_Image1_3.setItemText(2, _translate("MainWindow", "Median Filter"))
        self.comboBox_Image1_2.setItemText(0, _translate("MainWindow", "1"))
        self.comboBox_Image1_2.setItemText(1, _translate("MainWindow", "2"))
        self.comboBox_Image1_2.setItemText(2, _translate("MainWindow", "3"))
        self.label_4.setText(_translate("MainWindow", "Image "))
        self.label_2.setText(_translate("MainWindow", "Hybrid"))

