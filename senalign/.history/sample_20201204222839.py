# MyApp.py
# D. Thiebaut
# PyQt5 Application
# Editable UI version of the MVC application.
# Inherits from the Ui_MainWindow class defined in mainwindow.py.
# Provides functionality to the 3 interactive widgets (2 push-buttons,
# and 1 line-edit).
# The class maintains a reference to the model that implements the logic
# of the app.  The model is defined in class Model, in model.py.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot

    @pyqtSlot()
    def alignSlot(self):
        pass

    @pyqtSlot()
    def browseSlot(self):
        pass

    @pyqtSlot()
    def returnedPressedSlot(self):
        pass