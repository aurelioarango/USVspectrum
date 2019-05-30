
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap


def window():
    app = QApplication([])
    window = QWidget()
    window.setMinimumSize(850, 520)
    pixmap = QPixmap('episode.png')
    label = QLabel('Hello World!')
    label.setPixmap(pixmap)
    vbox = QVBoxLayout()
    vbox.addWidget(label)
    window.setWindowTitle("First Window")
    window.setLayout(vbox)
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    window()

"""

import PySide2
import sys
import random

import sys
import random
from PySide2 import QtCore, QtWidgets, QtGui


class MyWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "?????? ???"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        #self.text.pixmap()

        self.layout = QtWidgets.QVBoxLayout()
        self.text.pixmap =QtGui.QPixmap('episode.png')

        self.layout.addWidget(self.text.pixmap, 0,  QtWidgets)
        self.layout.add
        #self.text.pixmap(pixmap)
        #self.layout.addWidget(self.text)
        #self.layout.addWidget(self.button)
        #self.setLayout(self.layout)

        self.button.clicked.connect(self.magic)

    def magic(self):
        self.text.setText(random.choice(self.hello))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
"""