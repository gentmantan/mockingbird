#!/usr/bin/env python3
# from PyQt5.QtWidgets import (QAction, QApplication, QDialog, QLabel, QLineEdit,
#                              QMainWindow, QMenuBar, QPushButton, QVBoxLayout,
#                              QWidget, QDialogButtonBox, QMessageBox)
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal
from run_nlg import generate_text as nlg_gs
from run_markov import generate_text as markov_gs

class MyAPIOptionsBox(QDialog):
    values_selected = pyqtSignal(str, str)
    def __init__(self):
        super(MyAPIOptionsBox, self).__init__()

        # Set up the custom QDialog
        self.setWindowTitle("API Options")
        self.setGeometry(100, 100, 400, 200)
        self.setWindowIcon(QIcon('icon.png'))

        layout = QVBoxLayout()

        self.label1 = QLabel()
        # self.label1.setFont(QFont('Arial', 14))
        self.label1.setText("Consumer Key:")
        layout.addWidget(self.label1)

        self.text_input_1 = QLineEdit()
        layout.addWidget(self.text_input_1)

        self.label2 = QLabel()
        # self.label2.setFont(QFont('Arial', 14))
        self.label2.setText("Consumer Secret:")
        layout.addWidget(self.label2)

        self.text_input_2 = QLineEdit()
        layout.addWidget(self.text_input_2)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.accept)
        layout.addWidget(apply_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

        self.setLayout(layout)

    def accept(self):
        consumer_key = self.text_input_1.text()
        consumer_secret = self.text_input_2.text()
        self.values_selected.emit(consumer_key, consumer_secret)
        super(MyAPIOptionsBox, self).accept()

class MyScrollableMessageBox(QDialog):
    regenerate_button_clicked = pyqtSignal(str)
    def __init__(self, text):
        super(MyScrollableMessageBox, self).__init__()

        self.setWindowIcon(QIcon('icon.png'))

        # Set up the custom QDialog
        self.setWindowTitle("Generated Text")
        self.setGeometry(100, 100, 400, 200)

        # Create a QTextEdit widget
        text_edit = QTextEdit()
        # text_edit.setText("just setting up my twttr")
        text_edit.setPlainText(text)
        text_edit.setWordWrapMode(True)

        # Create a QScrollArea to hold the QTextEdit widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(text_edit)

        # Create buttons for OK and Cancel
        ok_button = QPushButton("Post")
        ok_button.clicked.connect(self.accept)
        regenerate_button = QPushButton("Regenerate")
        regenerate_button.clicked.connect(self.on_regenerate_button_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        # Create a layout for the custom QDialog
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.addWidget(ok_button)
        layout.addWidget(regenerate_button)
        layout.addWidget(cancel_button)



    def on_regenerate_button_clicked(self):
        # Emit the custom signal with a parameter
        self.regenerate_button_clicked.emit("Hello from Third Button!")


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.api_keys = []

        # Set up the main window
        self.setWindowTitle("Mockingbird UI")
        self.setGeometry(100, 100, 400, 200)
        self.setWindowIcon(QIcon('icon.png'))

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a vertical layout for the central widget
        layout = QVBoxLayout()

        self.label1 = QLabel()
        self.label1.setFont(QFont('Arial', 14))
        self.label1.setText("Please enter a seed text:")
        layout.addWidget(self.label1)

        # Create a text input box
        self.text_input = QLineEdit()
        layout.addWidget(self.text_input)

        # Create a submit button
        submit_button = QPushButton("Generate Text")
        submit_button.clicked.connect(self.on_submit)
        layout.addWidget(submit_button)

        # Set the layout for the central w
        central_widget.setLayout(layout)

        # Create a menu bar
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # Create a file menu
        file_menu = menu_bar.addMenu("File")

        # Create an action for the file menu
        action_exit = QAction("Exit", self)
        action_exit.triggered.connect(self.close)

        # Add the action to the file menu
        file_menu.addAction(action_exit)

        option_group = QActionGroup(self)

        options_menu = menu_bar.addMenu("Options")

        ## Model menu
        menu_model = QMenu("Model", self)
        options_menu.addMenu(menu_model)

        self.action_markov = QAction("Markov Chain", self)
        self.action_markov.setCheckable(True)
        self.action_markov.setChecked(True)
        self.action_markov.setActionGroup(option_group)
        self.action_markov.triggered.connect(self.update_action_rnn)
        menu_model.addAction(self.action_markov)

        self.action_rnn = QAction("RNN", self)
        self.action_rnn.setCheckable(True)
        self.action_rnn.setActionGroup(option_group)
        self.action_rnn.triggered.connect(self.update_action_markov)
        menu_model.addAction(self.action_rnn)

        ## Dataset menu
        menu_dataset = QMenu("Dataset", self)
        options_menu.addMenu(menu_dataset)

        action_twcs = QAction("Twitter Customer Service", self)
        action_twcs.setCheckable(True)
        action_twcs.setChecked(True)
        action_twcs.setActionGroup(option_group)
        menu_dataset.addAction(action_twcs)

        ## Markov options

        self.action_api = QAction("API Options", self)
        self.action_api.triggered.connect(self.update_action_api)
        options_menu.addAction(self.action_api)

        # action_ngen = QAction("# to Generate", self)
        # options_menu.addAction(action_ngen)

    def on_submit(self):
        seed_text = self.text_input.text()
        if self.action_markov.isChecked():
            print("Markov chosen")
            generated_text = markov_gs(seed_text)
        elif self.action_rnn.isChecked():
            print("NLG chosen")
            generated_text = nlg_gs('nlg_trained_model.h5', 'nlg_tokenizer.pkl', seed_text, 20)
        else:
            generated_text = "Error: text not generated"
        msg_box = MyScrollableMessageBox(generated_text)
        result = msg_box.exec_()

        if result == QDialog.Accepted:
            print("Post button clicked")

        elif result == QDialog.Rejected:
            print("Cancel button clicked")

    def update_action_rnn(self):
        self.action_markov.setChecked(True)
        self.action_rnn.setChecked(False)

    def update_action_markov(self):
        self.action_rnn.setChecked(True)
        self.action_markov.setChecked(False)


    def update_action_api(self):

        def get_api_keys(consumer_key, consumer_secret):
            self.api_keys.append(consumer_key)
            self.api_keys.append(consumer_secret)

        print("Selected action api")
        api_box = MyAPIOptionsBox()
        api_box.values_selected.connect(get_api_keys)
        result = api_box.exec_()
        if result == QDialog.Accepted:
            print(f"API settings applied")
            for key in self.api_keys:
                print(key)

        elif result == QDialog.Rejected:
            print("Canceled API settings")



if __name__ == "__main__":
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
