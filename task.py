# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'task.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1164, 771)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.comboBox_Image1 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Image1.setObjectName("comboBox_Image1")
        self.comboBox_Image1.addItem("")
        self.comboBox_Image1.addItem("")
        self.comboBox_Image1.addItem("")
        self.horizontalLayout.addWidget(self.comboBox_Image1)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.comboBox_Image1_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Image1_3.setObjectName("comboBox_Image1_3")
        self.comboBox_Image1_3.addItem("")
        self.comboBox_Image1_3.addItem("")
        self.comboBox_Image1_3.addItem("")
        self.horizontalLayout_2.addWidget(self.comboBox_Image1_3)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.comboBox_Image1_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_Image1_2.setObjectName("comboBox_Image1_2")
        self.comboBox_Image1_2.addItem("")
        self.comboBox_Image1_2.addItem("")
        self.comboBox_Image1_2.addItem("")
        self.horizontalLayout_3.addWidget(self.comboBox_Image1_2)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.image1 = QtWidgets.QLabel(self.centralwidget)
        self.image1.setText("")
        self.image1.setObjectName("image1")
        self.horizontalLayout_5.addWidget(self.image1)
        self.image2 = QtWidgets.QLabel(self.centralwidget)
        self.image2.setMinimumSize(QtCore.QSize(250, 250))
        self.image2.setText("")
        self.image2.setObjectName("image2")
        self.horizontalLayout_5.addWidget(self.image2)
        self.image3 = QtWidgets.QLabel(self.centralwidget)
        self.image3.setMinimumSize(QtCore.QSize(300, 250))
        self.image3.setText("")
        self.image3.setObjectName("image3")
        self.horizontalLayout_5.addWidget(self.image3)
        self.widget_2 = PlotWidget(self.centralwidget)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_5.addWidget(self.widget_2)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 5)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setObjectName("comboBox_3")
        self.horizontalLayout_6.addWidget(self.comboBox_3)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_7.addWidget(self.label_6)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_7.addWidget(self.comboBox_2)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_8.addWidget(self.label_7)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_8.addWidget(self.comboBox)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_9.addWidget(self.label_8)
        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.horizontalLayout_9.addWidget(self.comboBox_4)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_9)
        self.verticalLayout_2.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.widget_4 = PlotWidget(self.centralwidget)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_10.addWidget(self.widget_4)
        self.widget_5 = PlotWidget(self.centralwidget)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_10.addWidget(self.widget_5)
        self.widget = PlotWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.horizontalLayout_10.addWidget(self.widget)
        self.widget_3 = PlotWidget(self.centralwidget)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_10.addWidget(self.widget_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 5)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_12.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1164, 26))
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
        self.label.setText(_translate("MainWindow", "Original Image"))
        self.label_2.setText(_translate("MainWindow", "Add Noise"))
        self.comboBox_Image1.setItemText(0, _translate("MainWindow", "Add Noise"))
        self.comboBox_Image1.setItemText(1, _translate("MainWindow", "Gaussian Noise"))
        self.comboBox_Image1.setItemText(2, _translate("MainWindow", "Salt and Pepper Noise"))
        self.label_3.setText(_translate("MainWindow", "Remove Noise"))
        self.comboBox_Image1_3.setItemText(0, _translate("MainWindow", "Average Filter"))
        self.comboBox_Image1_3.setItemText(1, _translate("MainWindow", "Gaussian Filter"))
        self.comboBox_Image1_3.setItemText(2, _translate("MainWindow", "Median Filter"))
        self.label_4.setText(_translate("MainWindow", "Threshold"))
        self.comboBox_Image1_2.setItemText(0, _translate("MainWindow", "Choose threshold____"))
        self.comboBox_Image1_2.setItemText(1, _translate("MainWindow", "Global_threshold_v_127"))
        self.comboBox_Image1_2.setItemText(2, _translate("MainWindow", "Local threshold"))
        self.label_5.setText(_translate("MainWindow", "Edge detection"))
        self.label_6.setText(_translate("MainWindow", "Histogram"))
        self.label_7.setText(_translate("MainWindow", "Transforamtion"))
        self.comboBox.setItemText(0, _translate("MainWindow", "check transformation___"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Gray scale image"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Red channel histogram"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Green channel histogram"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Blue channel histogram"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Cumulative curve"))
        self.comboBox.setItemText(6, _translate("MainWindow", "Equalization histogram"))
        self.label_8.setText(_translate("MainWindow", "Hybrid Image"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "Hybrid_Images____"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Image 1"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "image 2"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "Hybrid image"))
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())