import os
import shutil
import subprocess
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QDesktopWidget, QVBoxLayout, QWidget, QLabel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("程序控制")
        self.resize(600, 200)

        # 设置窗口图标
        self.setWindowIcon(QIcon("images/VFD_Single_Logo.png"))

        # 获取屏幕的尺寸
        screen = QDesktopWidget().screenGeometry()

        # 计算窗口的左上角位置
        window_width = self.frameGeometry().width()
        window_height = self.frameGeometry().height()
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 3

        # 设置窗口的位置
        self.setGeometry(x, y, window_width, window_height)

        # 创建一个垂直布局
        layout = QVBoxLayout()

        # 创建按钮
        button = QPushButton("打开程序", self)
        button.clicked.connect(self.open_program)

        # 添加按钮到布局中
        layout.addWidget(button)

        # 创建说明文本
        label = QLabel("本程序在检测完人脸后会自动关闭，请重新打开程序")

        # 添加说明文本到布局中
        layout.addWidget(label)

        # 创建一个主部件，并将布局设置为主部件的布局
        widget = QWidget()
        widget.setLayout(layout)

        # 将主部件设置为窗口的中心部件
        self.setCentralWidget(widget)

    # 打开 GUI
    def open_program(self):
        program_path = "VFD_GUI.py"
        subprocess.Popen([sys.executable, program_path])


if __name__ == "__main__":
    folder_path = 'output'
    # 初始化临时存储文件夹
    if os.path.exists(folder_path):
        # 删除文件夹及其内容
        shutil.rmtree(folder_path)
    else:
        print("文件夹不存在，无需删除。")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
