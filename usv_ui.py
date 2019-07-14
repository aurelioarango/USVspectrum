"""Importing QT UI TOOLS"""
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, \
    QComboBox, QFileDialog,QListWidget, QSpacerItem, QAction, QMainWindow, QMessageBox, QDialog, QDialogButtonBox, \
    QLineEdit, QDoubleSpinBox, QSpinBox, QSizePolicy

from PyQt5.QtGui import QPixmap, QIcon, QTextLine
from PyQt5 import QtWidgets
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt

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

"""Install matlab engine for python
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

cd (fullfile(matlabroot,'extern','engines','python'))
system('python setup.py install')

or 
conda install matlab_engine


"""

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
        self.pixmap = QPixmap('unsplash_rat.jpg')


        #self.pixmap.scaled(1000,1000, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)

        self.previous_button = QPushButton('Previous')
        self.next_button = QPushButton('Next')

        self.previous_button.setStyleSheet(style)
        self.next_button.setStyleSheet(style)

        self.label.setPixmap(self.pixmap)
        self.label.setScaledContents(True)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        #self.setPalette(self.palet)
        self.resize(1000, 750)

        #verticalSpacer = QSpacerItem(200, 100, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.menu_bar()

        """"-------------- POSITION ------------------"""
        #self.label.move(500, 300)
        self.label.move(100,100)
        #self.list.move(100, 100)

        #self.combobox.setMaximumSize(100, 25)
        self.label.setMinimumSize(500,500)
        self.label.setMaximumSize(500,500)
        #self.label.pixmap().scaled(500,500,Qt.IgnoreAspectRatio,Qt.FastTransformation)

        #self.list.setMaximumSize(500,500)
        #self.load_button.setMaximumSize(100, 25)
        #self.start_button.setMaximumSize(100, 25)

        """"-------------- CONNECT BUTTON ------------------"""
        #self.load_button.clicked.connect(self.open_folder)
        #self.start_button.clicked.connect(self.process_file)

        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)


        """"-------------- ADD WIDGETS ------------------"""

        #

        #self.hlayout.addWidget(self.list)

        #self.hlayout.addWidget(self.label,Qt.AlignRight)
        #self.vlayout.addWidget(self.combobox)
        #self.vlayout.addWidget(self.load_button)
        #self.vlayout.addWidget(self.start_button)

        """"-------------- SET WINDOW ------------------"""
        self.setWindowTitle("USVSpectrum")
        self.setWindowOpacity(50)

        #self.list.setStyleSheet("background: grey;")

        #self.glayout.addWidget(self.list, 0, 0, 5, 3)
        # self.glayout.addItem(verticalSpacer)
        self.glayout.addWidget(self.label, 0, 5, 5, 5)

        #self.glayout.addWidget(self.combobox, 9, 0)

        """Buttons Postions"""

        #self.glayout.addWidget(self.load_button, 9, 2)
        #self.glayout.addWidget(self.start_button, 9, 3)
        self.glayout.addWidget(self.previous_button, 9, 6)
        self.glayout.addWidget(self.next_button, 9, 8)

        self.spacer = QSpacerItem(20,20)
        self.glayout.addItem(self.spacer, 10, 0)
        self.glayout.addItem(self.spacer, 0, 11, 1, 5)
        self.glayout.addItem(self.spacer,0,0,1,3)
        #self.glayout.addItem()

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
        extracAction = QAction(QIcon('new.png'), 'Extract Images', self)

        """"-------------- CONNECT MENU ITEMS ------------------"""
        selectSurceAction.triggered.connect(self.select_view_folder)
        extracAction.triggered.connect(self.show_image_extract_dialog)

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
        filemenu.addAction(extracAction)
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


    def done_training(self):
        QMessageBox.about(self, 'Warning', 'Done Training')

    def train_model(self):
        #print("training model")

        self.show_train_dialog()

        """ Select model properties"""


        #load_training_data_path = QFileDialog.getExistingDirectory(self, 'Load Data Directory', self.path_classified,QFileDialog.DontResolveSymlinks)
        #load_training_data_path = self.path_classified
        if self.model_run_status:
            save_model_path = self.path_models
        #    print(save_model_path)
        #    epochs = 1
        #    print(epochs)
        #    """Pass in classification folder"""
        #    """Pass the path to where to save the model"""
        #    """Return to working directory"""
        #    """Using Threads to avoid GUI freezing"""
            train_thread = threading.Thread(target= transfer_learning.main, args=(self.path_classified,
                                self.selected_model_name, self.model_name, self.epochs, self.learning_rate, self.path_models))
            train_thread.start()



        #QMessageBox(self,"Warning", "Finished Training Model")


        #os.chdir(self.path)


    def retrain_model(self):
        print("retraining model")


    def classify_data(self):
        """ Sorting"""

        # print('Models: ',self.path_models)
        # print('TestExtracted data: ', self.path_extracted)
        #selected_model =''
        self.show_classify_dialog()
        """Which Model to load """
        #self.selected_model = QFileDialog.getOpenFileName(self, 'Select Model', self.path_models, 'Model ( *.pth )')
        #self.selected_model
        """Create thread to classify or UI wil freeze"""
        if self.classify_status:
            classify_thread = threading.Thread(target=pytorch_classify.main, args= (self.path_extracted, self.selected_model, self.path_output_classified) )
            #pytorch_classify.main(self.path_extracted, selected_model[0], self.path_output_classified)
            classify_thread.start()
        else:
            print("Cancelled")

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
            self.path_audio = self.path + '\\audio'
            self.path_matlab_script = self.path_scripts+"\\usv_matlab\\"
        else:
            self.path_images = self.path+'/usv_images'
            self.path_extracted = self.path+'/usv_images/extracted/'
            self.path_classified = self.path+'/usv_images/classified'
            self.path_output_classified = self.path + '/output'
            self.path_models = self.path+'/usv_models'
            self.path_scripts = self.path+'/scripts'
            self.path_audio = self.path + '/audio'
            self.path_matlab_script = self.path_scripts + "/usv_matlab/"

        """Verify And/Or Create Directories"""
        pathlib.Path(self.path_images).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_extracted).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_classified).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_models).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_audio).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path_matlab_script).mkdir(parents=True, exist_ok=True)

    def show_image_extract_dialog(self):
        """Create dialog to extract data"""
        print("Show dialog")
        self.extract_dialog = QDialog()
        self.extract_dialog.setWindowTitle("Extract Images Window")
        self.extract_dialog.setMaximumSize(600, 300)
        self.extract_dialog.setMinimumSize(600, 300)

        self.extract_dialog_list = QListWidget(self.extract_dialog)
        self.extract_dialog_list.setMinimumSize(560,200)
        self.extract_dialog_combobox = QComboBox(self.extract_dialog)
        self.extract_dialog_dir_line = QLineEdit(self.path_audio,self.extract_dialog)

        self.extract_dialog_select_dir_button = QPushButton('Directory',self.extract_dialog)
        self.extract_dialog_extract_button = QPushButton('Extract', self.extract_dialog)
        self.extract_dialog_cancel_button = QPushButton('Cancel', self.extract_dialog)

        """Element Position"""
        self.extract_dialog_select_dir_button.move(15,15)
        self.extract_dialog_dir_line.move(110, 20)
        self.extract_dialog_list.move(20,50)
        self.extract_dialog_combobox.move(250,265)
        self.extract_dialog_cancel_button.move(15,265)
        self.extract_dialog_extract_button.move(500,265)

        """Set up"""
        #self.extract_dialog_dir_line.text()
        self.extract_dialog_dir_line.setMinimumSize(470,20)
        self.extract_dialog_combobox.addItems(["Single","Multiple", "All"])

        """Connect combo box"""
        self.extract_dialog_select_dir_button.clicked.connect(self.extract_select_directory)
        self.extract_dialog_extract_button.clicked.connect(self.extract_images_from_file)

        self.extract_dialog.exec()
    def extract_select_directory(self):

        self.fnames = QFileDialog.getOpenFileNames(self,'Open Folder', self.path, "USV file(s) (*.wav *WAV)")
        #fname = QFileDialog.getOpenFileName(self, 'Open file(s)', current_path, "USV file(s) (*.raw)")
        if self.fnames:
            self.extract_dialog_list.clear()
            #print(len(self.fnames))
            for i in self.fnames[0]:
                self.extract_dialog_list.addItem(i)
            """Select Audio Diectory"""
            print(self.extract_dialog_list)

    def extract_images_from_file(self):

        if self.fnames != None :

            #print(self.combobox.currentIndex())
            os.chdir(self.path_matlab_script)
            """---------------------------- Change Path To M file--------------------------------"""
            print(os.getcwd())
            """----------------------------------- Run All USV Files From A Single Folder"""
            #if self.extract_dialog_combobox.currentIndex() == 0:
            if self.extract_dialog_combobox.currentText() == "Single":
                #for x in self.fnames[0]:
                path = self.extract_dialog_list.currentIndex().data()+""
                last_of_index = path.rfind("/")
                folder_path = path[:last_of_index]
                print(folder_path)
                command = "process_rusv(\"" + folder_path + "\",\"all\");exit;"

                #subprocess.run(["matlab", "-nodesktop", "-nosplash", "-r", command])
                #print(x)

            #"""-----------------Implement Single Case in Matlab------------------------"""

            #elif self.extract_dialog_combobox.currentIndex() == 1:
            elif self.extract_dialog_combobox.currentText() == "Mutiple":
                path = self.extract_dialog_list.currentIndex().data()
                last_of_index = path.rfind("/")
                file_name = path[last_of_index+1:]
                folder_path = path[:last_of_index]

                command = "process_rusv(\"" + folder_path + "\",\"" + file_name+"\" );exit;"
                print(command)
                #subprocess.run(["matlab", "-nodesktop", "-nosplash", "-r",command])
                print(self.extract_dialog_list.currentIndex().data())

            elif self.extract_dialog_combobox.currentText() == "All":
                """"""
                print("All")
                list_audio_files =[]
                #print(self.extract_dialog_list.items())
                for i in range (self.extract_dialog_list.count()):
                    """Add items to te list"""
                    #print(self.)

                    list_audio_files.append(self.extract_dialog_list.item(i).text())


                print(list_audio_files)

            os.chdir(self.path)
            print(os.getcwd())

    def show_train_dialog(self):
        """Create Train Dialog"""
        self.train_dialog = QDialog()
        self.train_dialog.setWindowTitle("Training Window")
        self.train_dialog.setMaximumSize(600,300)
        self.train_dialog.setMinimumSize(600,300)
        """Drop Menu"""
        self.model_dropmenu = QComboBox(self.train_dialog)
        self.model_dropmenu.move(15,50)
        self.model_dropmenu.addItems(['resnet18','resnet34','resnet50','resnet101', 'resnet152','resnext50_32x4d',
                                     'vgg19','alexnet','squeezenet','densenet','googlenet','inception'])
        self.model_dropmenu.currentIndexChanged.connect(self.model_index_change)

        cancel_button = QPushButton("Cancel", self.train_dialog)
        run_button = QPushButton("Run", self.train_dialog)

        cancel_button.move(15, 250)
        run_button.move(500, 250)

        source_button = QPushButton("Data Directory:", self.train_dialog)
        source_button.move(11, 90)
        source_button.clicked.connect(self.set_training_data)
        self.source_training_data = QLineEdit(self.path_classified, self.train_dialog)
        self.source_training_data.move(170,95)
        self.source_training_data.setMinimumSize(400,20)
        self.source_training_data.setMaximumSize(400,20)



        """Setup of Learning Rate"""
        self.source_learning_rate = QDoubleSpinBox(self.train_dialog)
        self.source_learning_rate.move(300,50)
        self.source_learning_rate.setDecimals(6)
        self.source_learning_rate.setSingleStep(0.0001)
        self.source_learning_rate.setRange(0.00001,1)
        self.source_learning_rate.setValue(0.0001)
        self.source_learning_rate.valueChanged.connect(self.model_learning_rate_change)

        learning_rate_label = QLabel(self.train_dialog)
        learning_rate_label.setText("Learning Rate: ")
        learning_rate_label.move(200,54)
        """Setup of Number of Epochs"""
        self.source_epochs = QSpinBox(self.train_dialog)
        self.source_epochs.move(500,50)
        self.source_epochs.setSingleStep(10)
        self.source_epochs.setRange(1,1000)
        self.source_epochs.setValue(10)
        self.source_epochs.valueChanged.connect(self.model_epochs_change)

        epochs_label = QLabel(self.train_dialog)
        epochs_label.move(450,54)
        epochs_label.setText("Epochs: ")

        """Setting up Model Name"""

        self.source_model_name = QLineEdit('resnet18',self.train_dialog)
        self.source_model_name.move(170,140)
        self.source_model_name.setMaximumSize(400,20)
        self.source_model_name.setMinimumSize(400,20)
        self.source_model_name.textChanged.connect(self.model_name_text_change)

        model_name_label = QLabel(self.train_dialog,text="Model Name: ")
        model_name_label.move(20,144)
        """Where to save model"""

        model_save_directory_button = QPushButton("Save Directory",self.train_dialog)
        model_save_directory_button.move(15, 190)
        model_save_directory_button.clicked.connect(self.model_save_directory)


        self.source_save_training_directory =QLineEdit(self.path_models,self.train_dialog)
        self.source_save_training_directory.move(170, 190)
        self.source_save_training_directory.setMinimumSize(400, 20)
        self.source_save_training_directory.setMaximumSize(400, 20)

        cancel_button.clicked.connect(self.model_cancel)
        run_button.clicked.connect(self.model_run)

        self.learning_rate = 0.0001
        self.epochs=10
        self.model_run_status = False
        self.selected_model_name = 'resnet18'
        self.model_name ='resnet18'
        self.train_dialog.exec_()


    def model_cancel(self):
        self.train_dialog.close()
        self.model_run_status = False

    def model_run(self):
        self.train_dialog.close()
        if self.source_model_name.text() != "":
            self.model_run_status = True
        else:
            self.model_run_status = False

    def model_save_directory(self):

        path_outdir = QFileDialog.getExistingDirectory(self, 'Data Directory', self.path_output_classified,
                                                        QFileDialog.DontResolveSymlinks)
        if path_outdir:
            self.path_models = self.source_save_training_directory.text()


    def set_training_data(self):
        """ Set source directory to train data """
        path_outdir = QFileDialog.getExistingDirectory(self, 'Model Directory', self.path_output_classified,
                                                       QFileDialog.DontResolveSymlinks)
        if path_outdir:
            self.path_classified = self.source_training_data.text()
        #print(self.path_classified)
    def model_name_text_change(self):

        self.model_name = self.source_model_name.text()

        #print(self.model_name)

    def model_epochs_change(self):
        """Retrieving Number of Epochs"""
        self.epochs =self.source_epochs.value()
        #print(self.epochs)
    def model_learning_rate_change(self):
        """Retrieving  learning rate"""
        self.learning_rate = self.source_learning_rate.value()
        #print(self.learning_rate)


    def model_index_change(self,index):
        """Get model name"""
        self.selected_model_name = self.model_dropmenu.currentText()
        self.source_model_name.setText(self.selected_model_name)

        #print(self.selected_model_name)

    def show_classify_dialog(self):
        """Show Image dialog"""
        self.classify_dialog = QDialog()
        #dialog.setStyleSheet("background: grey;")
        self.classify_dialog.setWindowTitle("Classify Window")
        self.classify_dialog.setMinimumSize(600,300)
        self.classify_dialog.setMaximumSize(600,300)


        cancel_button = QPushButton("Cancel",self.classify_dialog)
        apply_button = QPushButton("Apply", self.classify_dialog)
        #set_button = QPushButton(QtWidgets.QDialogButtonBox.Apply)
        cancel_button.move(15,250)
        apply_button.move(500,250)

        cancel_button.clicked.connect(self.cancel_classify)
        apply_button.clicked.connect(self.apply_classify)

        model_button = QPushButton("Model: ",self.classify_dialog)
        model_button.move(15,95)
        model_button.clicked.connect(self.getmodel)
        self.source_model = QLineEdit(self.classify_dialog)
        self.source_model.move(170, 100)

        self.source_model.setMinimumSize(400,20)
        self.source_model.setMaximumSize(400, 20)

        dataout_button =QPushButton("Output Directory: ", self.classify_dialog)
        dataout_button.move(15,145)
        dataout_button.clicked.connect(self.getoutdata)
        self.source_dataout = QLineEdit(self.classify_dialog)
        self.source_dataout.move(170,150 )
        self.source_dataout.setMinimumSize(400, 20)
        self.source_dataout.setMaximumSize(400, 20)

        source_button = QPushButton("Data Directory:", self.classify_dialog)
        source_button.move(15,195)
        source_button.clicked.connect(self.getindata)
        self.source_datain = QLineEdit(self.classify_dialog)
        self.source_datain.move(170,195)
        self.source_datain.setMinimumSize(400, 20)
        self.source_datain.setMaximumSize(400, 20)

        try:

            self.source_model.setText(self.path_models)
            if self.source_model.text().find(".pth") == -1 :
                self.source_model.setStyleSheet("color:red")
                print("Color red")
            else:
                self.source_model.setStyleSheet("color:black")
                print("Color black")

            self.source_datain.setText(self.path_classified)
            self.source_dataout.setText(self.path_output_classified)
        except:
            print("Cannot Set Field Values")

        self.classify_status = False
        # dialog.setWindowModality(Qt.ApplicationModal)
        self.classify_dialog.exec_()

        #set_button.acc
    def apply_classify(self):
        self.classify_dialog.close()
        self.classify_status= True

    def cancel_classify(self):
        self.classify_dialog.close()
        self.classify_status = False

    def getoutdata(self):
        """Get data from line edit"""
        #self.source_dataout.text()
        path_outdir=QFileDialog.getExistingDirectory(self, 'Output Directory', self.path_output_classified,QFileDialog.DontResolveSymlinks)
        if path_outdir:
            self.source_datain.setText(path_outdir)
            #self.path_output_classified = path_outdir

    def getindata(self):
        """ """
        path_indata = QFileDialog.getExistingDirectory(self, 'Source Data', self.path_classified,QFileDialog.DontResolveSymlinks)
        if path_indata:
            self.source_dataout.setText(path_indata)
            #self.path_extracted = path_indata

    def getmodel(self):
        """Getting model"""
        model_path, ok = QFileDialog.getOpenFileName(self, 'Open Model', self.path_models, "Models (*.pth)")

        if ok:
            self.source_model.setText(model_path)
            #self.selected_model = model_path
        self.selected_model = model_path


        if self.source_model.text().find(".pth") == -1:
            self.source_model.setStyleSheet("color:red")
            print("Color red")
        else:
            self.source_model.setStyleSheet("color:black")
            print("Color black")

    def model_apply(self):
        if self.temp_model_path and self.temp_path_indata and self.temp_path_outdir:
            self.selected_model = self.temp_model_path
            self.path_extracted = self.temp_path_indata
            self.path_output_classified = self.temp_path_outdir
        else:
            print("Error while setting directory outputs")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    execute = usv_gui()
    execute.show()
    app.exec_()

    #load_window(app)
