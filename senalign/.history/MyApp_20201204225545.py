# MyApp.py
# D. Thiebaut
# PyQt5 Application
# Editable UI version of the MVC application.
# Inherits from the Ui_MainWindow class defined in mainwindow.py.
# Provides functionality to the 3 interactive widgets (2 push-buttons,
# and 1 line-edit).
# The class maintains a reference to the file that implements the logic
# of the app.  The file is defined in class file, in file.py.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from mainwindow import Ui_MainWindow
import sys
import logging
from file import File


class MainWindowUIClass(Ui_MainWindow):
    def __init__(self):
        '''Initialize the super class
        '''
        super().__init__()
        self.file = File()

    def setupUi(self, MW):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi(MW)

    def debugPrint(self, msg):
        '''Print the message in the text edit at the bottom of the
        horizontal splitter.
        '''
        self.debugTextBrowser.append(msg)

    def refreshAll(self):
        '''
        Updates the widgets whenever an interaction happens.
        Typically some interaction takes place, the UI responds,
        and informs the file of the change.  Then this method
        is called, pulling from the file information that is
        updated in the GUI.
        '''
        self.lineEdit.setText(self.file.getFileName())

    # slot
    def returnedPressedSlot(self):
        ''' Called when the user enters a string in the line edit and
        presses the ENTER key.
        '''
        fileName = self.lineEdit.text()
        if self.file.isValid(fileName):
            self.file.setFileName(self.lineEdit.text())
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid file name!\n" + fileName)
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEdit.setText("")
            self.refreshAll()
            self.debugPrint("Invalid file specified: " + fileName)

    # slot
    def alignSlot(self):
        ''' Called when the user presses the Align-Doc button.
        '''
        pass
        # self.debugPrint("Align-Doc button pressed!")

    # slot
    def browseSlot(self):
        ''' Called when the user presses the Browse button
        '''
        # self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;Word Document (*.docx)",
            options=options)
        if fileName:
            logging.info("Choosen file: " + fileName)
            self.debugPrint("Choosen file: " + fileName)
            self.file.setFileName(fileName)
            self.refreshAll()


def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.

    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


main()
