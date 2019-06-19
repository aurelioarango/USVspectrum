"""Importing QT UI TOOLS"""
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, \
    QComboBox, QFileDialog,QListWidget, QSpacerItem, QAction, QMainWindow, QMessageBox

from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtWidgets
import PyQt5.QtCore as QtCore

"""Importing System Tools"""

import os, sys
import subprocess
import pathlib
import threading
import platform

"""Importing Pytorch Training models"""

sys.path.append(os.getcwd())
if platform == "Windows":
    sys.path.append(os.getcwd() + "\\scripts\\pytorch")
    sys.path.append(os.getcwd() + "\\scripts\\classify")
    sys.path.append(os.getcwd() + "\\scripts\\usv_sort")
else:
    sys.path.append(os.getcwd() + "/scripts/pytorch")
    sys.path.append(os.getcwd() + "/scripts/classify")
    sys.path.append(os.getcwd() + "/scripts/usv_sort")

import transfer_learning
import pytorch_classify
import glob

class usv_gui (QMainWindow):
    def __init__(self, parent=None):
        """--------------- ELEMENTS ------------------"""
        super(usv_gui, self).__init__(parent)

        #self.form_widget = QFormLayout(self)
        #QMainWindow.__init__(self)
        #self.setStyle(QStyleFactory.create('Fusion'))
        #print(QStyleFactory.keys())
        style = "QPushButton { background-color: #F9AA33; border-style: outset; padding: 6px; border-width: 2px; border-color: beige }"
        #self.setStyleSheet("QPushButton { background-color: yellow; border-style: outset; padding: 6px; border-width: 2px; border-color: beige }")
        #self.palet = QPalette()

        #self.palet.setColor(QPalette.Background,QColor(150,0,0) )
        self.widget = QWidget()

        self.setCentralWidget(self.widget)
        self.centralWidget().setStyleSheet("background: grey")

        self.current_image=0
        self.view_images=[]

        self.vlayout = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.glayout = QGridLayout()
        self.label = QLabel()
        self.fnames=None
        pixmap = QPixmap('test_image.png')


        self.load_button = QPushButton('Load')
        self.start_button = QPushButton('Start')
        self.previous_button = QPushButton('Previous')
        self.next_button = QPushButton('Next')

        self.load_button.setStyleSheet(style)
        self.start_button.setStyleSheet(style)
        self.previous_button.setStyleSheet(style)
        self.next_button.setStyleSheet(style)


        self.list = QListWidget()
        self.combobox = QComboBox()
        self.label.setPixmap(pixmap)
        #self.setPalette(self.palet)
        self.resize(1000, 750)

        #verticalSpacer = QSpacerItem(200, 100, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.menu_bar()

        """"-------------- POSITION ------------------"""
        #self.label.move(500, 300)
        self.label.move(100,100)
        self.list.move(100, 100)

        self.combobox.setMaximumSize(100, 25)
        self.label.setMinimumSize(500,500)

        self.list.setMaximumSize(500,500)
        self.load_button.setMaximumSize(100, 25)
        self.start_button.setMaximumSize(100, 25)

        """"-------------- CONNECT BUTTON ------------------"""
        self.load_button.clicked.connect(self.open_folder)
        self.start_button.clicked.connect(self.process_file)

        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)


        """"-------------- ADD WIDGETS ------------------"""

        self.combobox.addItems(["Multiple", "Single"])

        #self.hlayout.addWidget(self.list)

        #self.hlayout.addWidget(self.label,Qt.AlignRight)
        #self.vlayout.addWidget(self.combobox)
        #self.vlayout.addWidget(self.load_button)
        #self.vlayout.addWidget(self.start_button)

        """"-------------- SET WINDOW ------------------"""
        self.setWindowTitle("USVSpectrum")
        self.setWindowOpacity(50)

        self.list.setStyleSheet("background: grey;")

        self.glayout.addWidget(self.list, 0, 0, 5, 5)
        #self.glayout.addItem(verticalSpacer)
        self.glayout.addWidget(self.label, 0, 5, 5, 5 )

        self.glayout.addWidget(self.combobox,9,0 )

        """Buttons Postions"""

        self.glayout.addWidget(self.load_button, 9, 2)
        self.glayout.addWidget(self.start_button, 9, 3)
        self.glayout.addWidget(self.previous_button, 9, 6)
        self.glayout.addWidget(self.next_button, 9, 8)

        self.spacer = QSpacerItem(20,20)
        self.glayout.addItem(self.spacer,10,0 )

        """Need to add other buttons"""

        """--------------------------------Adding place holder for Stats to the Screen------------------------------"""
        label_detection = QLabel("Detected: ")
        label_frequency = QLabel("Average Freq: ")
        label_duration = QLabel("Duration: ")
        """Adding stats label"""
        #self.glayout.addWidget(label_detection, 1, 1)
        #self.glayout.addWidget(label_frequency, 2, 1)
        #self.glayout.addWidget(label_duration, 3, 1)


        self.centralWidget().setLayout(self.glayout)
        #self.setLayout(self.glayout)
        self.setMaximumSize(1000, 750)
        self.setMinimumSize(1000, 750)

        """SETUP ENVIRONMENT"""
        self.setup_environment()

        #app.exec_()

    def menu_bar(self):

        """"-------------- Creating menu actions------------------"""
        trainAction = QAction(QIcon('new.png'),"Train Model", self)
        retrainAction = QAction(QIcon('new.png'),"Retrain Model",self)

        classifyAction = QAction(QIcon('new.png'), "Classify Data",self)

        exitAction = QAction(QIcon("exit24.png"), "Exit",self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit Application")

        selectSurceAction = QAction(QIcon('new.png'), 'Open View Source', self)

        """"-------------- CONNECT MENU ITEMS ------------------"""
        selectSurceAction.triggered.connect(self.select_view_folder)

        trainAction.triggered.connect(self.train_model)
        retrainAction.triggered.connect(self.retrain_model)
        exitAction.triggered.connect(self.close)

        classifyAction.triggered.connect(self.classify_data)


        self.main_menu = self.menuBar()
        filemenu= self.main_menu.addMenu("&File")
        trainmenu = self.main_menu.addMenu("&Train")
        classify_menu = self.main_menu.addMenu("&Classify")


        #trainmenu.addMenu("&TrainModel")
        """"-------------- ADDING MENU ACTIONS ------------------"""
        """Adding to File Menu"""
        filemenu.addAction(exitAction)
        filemenu.addAction(selectSurceAction)
        """Adding to Train Menu"""
        trainmenu.addAction(trainAction)
        trainmenu.addAction(retrainAction)
        """Adding to Classify Menu"""
        classify_menu.addAction(classifyAction)

    def keyPressEvent(self,event):
        key = event.key()
        print(key)
        print(QtCore.Qt.RightArrow)

        """Changed to correct call value, not hardcoded """
        if key == 16777236:
        #if key == QtCore.Qt.RightArrow:

            self.next_image()
        elif key == 16777234:
        #elif key == QtCore.Qt.LeftArrow:
            self.previous_image()

    def select_view_folder(self):
        """Select a source folder to display images"""

        #self.view_images = QFileDialog.getOpenFileNames(self, 'Open Folder', self.main_path)
        path = QFileDialog.getExistingDirectory(self, 'Viewing Directory', self.path_extracted, QFileDialog.DontResolveSymlinks)
        self.current_image = 0


        list_of_images = glob.glob(path+'/*.png')

        #print(list_of_images)
        self.view_images = list_of_images
        if len (self.view_images) > 0:
            self.label.setPixmap( QPixmap(self.view_images[0]))
            #print(self.view_images[0])


    def next_image(self):
        """Select next image in foler"""
        #print("next image")
        if self.current_image < len(self.view_images):
            self.current_image = self.current_image+1
            self.label.setPixmap(QPixmap(self.view_images[self.current_image]))



    def previous_image(self):
        """Select previous image in folder"""
        #print("previous image")
        if self.current_image > 0 and self.current_image != 0:
            self.current_image = self.current_image - 1

            self.label.setPixmap(QPixmap(self.view_images[self.current_image]))

    def open_folder(self):


        self.fnames = QFileDialog.getOpenFileNames(self,'Open Folder', self.path)
        #fname = QFileDialog.getOpenFileName(self, 'Open file(s)', current_path, "USV file(s) (*.raw)")
        self.list.clear()
        #print(len(self.fnames))
        for i in self.fnames[0]:
            self.list.addItem(i)

    def process_file(self):

        if self.fnames != None :

            #print(self.combobox.currentIndex())
            os.chdir("/mnt/5442922B4292123A/Users/Aurelio/My Documents/MATLAB/mupet/usv_m/")
            """---------------------------- Change Path To M file--------------------------------"""
            print(os.getcwd())
            """----------------------------------- Run All USV Files From A Single Folder"""
            if self.combobox.currentIndex() == 0:

                #for x in self.fnames[0]:
                path = self.list.currentIndex().data()+""
                last_of_index = path.rfind("/")
                folder_path = path[:last_of_index]
                print(folder_path)
                command = "process_rusv(\"" + folder_path + "\",\"all\");exit;"

                subprocess.run(["matlab", "-nodesktop", "-nosplash", "-r", command])
                #print(x)

            #"""-----------------Implement Single Case in Matlab------------------------"""

            elif self.combobox.currentIndex() == 1:

                path = self.list.currentIndex().data()
                last_of_index = path.rfind("/")
                file_name = path[last_of_index+1:]
                folder_path = path[:last_of_index]

                command = "process_rusv(\"" + folder_path + "\",\"" + file_name+"\" );exit;"
                print(command)
                subprocess.run(["matlab", "-nodesktop", "-nosplash", "-r",command])
                print(self.list.currentIndex().data())

            os.chdir(self.path)
            print(os.getcwd())
    def done_training(self):
        QMessageBox.about(self, 'Warning', 'Done Training')

    def train_model(self):
        #print("training model")


        """ Select model properties"""
        model = 'resnet18' # can pick from any resnet, vgg, etc

        load_training_data_path = QFileDialog.getExistingDirectory(self, 'Load Data Directory', self.path_classified,QFileDialog.DontResolveSymlinks)
        #load_training_data_path = self.path_classified
        if load_training_data_path:
            save_model_path = self.path_models
            print(save_model_path)
            epochs = 1
            print(epochs)
            """Pass in classification folder"""
            """Pass the path to where to save the model"""
            """Return to working directory"""
            """Using Threads to avoid GUI freezing"""
            train_thread = threading.Thread(target= transfer_learning.main, args=(load_training_data_path,model, epochs,
                                                                                  0.0001,save_model_path) )
            train_thread.start()



        #QMessageBox(self,"Warning", "Finished Training Model")


        #os.chdir(self.path)


    def retrain_model(self):
        print("retraining model")


    def classify_data(self):
        """ Sorting"""

        # print('Models: ',self.path_models)
        # print('TestExtracted data: ', self.path_extracted)
        """Which Model to load """
        selected_model = QFileDialog.getOpenFileName(self, 'Select Model', self.path_models,'*.pth')
        """Create thread to classify or UI wil freeze"""
        classify_thread = threading.Thread(target=pytorch_classify.main, args= (self.path_extracted, selected_model[0], self.path_output_classified) )
        #pytorch_classify.main(self.path_extracted, selected_model[0], self.path_output_classified)
        classify_thread.start()


    def setup_environment(self):
        """Setting up the working space"""
        self.path = os.getcwd()
        # print("path: ", self.path)
        if platform.system() == "Windows":
            self.path_images = self.path + '\\usv_images'
            self.path_extracted = self.path + '\\usv_images\\extracted\\'
            self.path_classified = self.path + '\\usv_images\\classified'
            self.path_output_classified = self.path + '\\output'
            self.path_models = self.path + '\\usv_models'
            self.path_scripts = self.path + '\\scripts'
        else:
            self.path_images = self.path+'/usv_images'
            self.path_extracted = self.path+'/usv_images/extracted/'
            self.path_classified = self.path+'/usv_images/classified'
            self.path_output_classified = self.path + '/output'
            self.path_models = self.path+'/usv_models'
            self.path_scripts = self.path+'/scripts'

        """Verify And/Or Create Directories"""
        pathlib.Path(self.path_images).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_extracted).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_classified).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_models).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    execute = usv_gui()
    execute.show()
    app.exec_()

    #load_window(app)
