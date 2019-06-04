"""Importing QT UI TOOLS"""
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, \
    QComboBox, QFileDialog,QListWidget, QSpacerItem, QSizePolicy, QAction, QMainWindow, QFormLayout, QStyleFactory, \
    QMessageBox

from PyQt5.QtGui import QPixmap, QPalette, QIcon, QColor
from PyQt5 import QtWidgets, QtQuick

"""Importing System Tools"""

import os, sys
import subprocess
import pathlib
import threading

"""Importing Pytorch Training models"""

sys.path.append(os.getcwd())

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


        self.vlayout = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.glayout = QGridLayout()
        self.label = QLabel()
        self.fnames=None
        pixmap = QPixmap('test_image.png')
        self.load_button = QPushButton('Load')
        self.load_button.setStyleSheet(style)

        self.start_button = QPushButton('Start')
        self.list = QListWidget()
        self.combobox = QComboBox()
        self.label.setPixmap(pixmap)
        #self.setPalette(self.palet)
        self.resize(1000, 750)

        verticalSpacer = QSpacerItem(200, 100, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.menu_bar()

        """"-------------- POSITION ------------------"""

        self.combobox.setMaximumSize(100, 25)
        self.label.setMinimumSize(500,500)
        self.label.move(500,300)
        self.list.move(100,200)
        self.list.setMaximumSize(400,500)
        self.load_button.setMaximumSize(100, 25)
        self.start_button.setMaximumSize(100, 25)

        """"-------------- CONNECT BUTTON ------------------"""
        self.load_button.clicked.connect(self.open_folder)
        self.start_button.clicked.connect(self.process_file)

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

        self.glayout.addWidget(self.list, 0, 0)
        #self.glayout.addItem(verticalSpacer)
        self.glayout.addWidget(self.label, 0, 2)
        #self.glayout.addLayout(self.hlayout,0,0)
        self.glayout.addWidget(self.combobox,2,0 )
        self.glayout.addWidget(self.load_button, 1, 0)
        self.glayout.addWidget(self.start_button, 3, 0)

        """--------------------------------Adding place holder for Stats to the Screen------------------------------"""
        label_detection = QLabel("Detected: ")
        label_frequency = QLabel("Average Freq: ")
        label_duration = QLabel("Duration: ")
        self.glayout.addWidget(label_detection, 1, 1)
        self.glayout.addWidget(label_frequency, 2, 1)
        self.glayout.addWidget(label_duration, 3, 1)


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

        """"-------------- CONNECT MENU ITEMS ------------------"""
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
        filemenu.addAction(exitAction)
        trainmenu.addAction(trainAction)
        trainmenu.addAction(retrainAction)
        classify_menu.addAction(classifyAction)




    def open_folder(self):
        self.main_path = os.getcwd()

        self.fnames =QFileDialog.getOpenFileNames(self,'Open Folder', self.main_path)
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

            os.chdir(self.main_path)
            print(os.getcwd())
    def done_training(self):
        QMessageBox.about(self, 'Warning', 'Done Training')

    def train_model(self):
        print("training model")
        """change to scripts path"""
        #os.chdir(self.path_scripts)

        """Call Fastai/pytorch script"""
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')
        #print(device)

        """ Select model properties"""
        model = 'resnet18' # can pick from any restnet, vgg, etc
        load_training_data_path = self.path_classified
        save_model_path = self.path_models
        print(save_model_path)
        epochs = 1
        print(epochs)
        """Pass in classification folder"""
        """Pass the path to where to save the model"""
        """Return to working directory"""
        """Using Threads to avoid GUI freezing"""
        train_thread = threading.Thread(target= transfer_learning.main, args=(load_training_data_path,model, 10,
                                                                              0.0001,save_model_path) )
        train_thread.start()

        #QMessageBox(self,"Warning", "Finished Training Model")


        #os.chdir(self.path)



    def retrain_model(self):
        print("retraining model")


    def classify_data(self):
        """ Sorting"""

        print('Models: ',self.path_models)
        print('TestExtracted data: ', self.path_extracted)
        list_of_models = glob.glob(self.path_models+"/*")
        try:

            latest_model = max(list_of_models, key=os.path.getctime)
            print(latest_model)
        except ValueError:
            print("No Models in directory")
        else:
            print("ERROR: While locating model")

        #pytorch_classify.main(self.path_extracted, self.path_models)

    def setup_environment(self):
        """Setting up the working space"""
        self.path = os.getcwd()
        self.path_images = self.path+'/usv_images'
        #print("path: ", self.path)
        self.path_extracted = self.path+'/usv_images/extracted'
        self.path_classified = self.path+'/usv_images/classified'
        self.path_models = self.path+'/usv_models'
        self.path_scripts = self.path+'/scripts'
        """Verify And/Or Create Directories"""
        pathlib.Path(self.path_images).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_extracted).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_classified).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_models).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app
    execute = usv_gui()
    execute.show()
    app.exec_()

    #load_window(app)
