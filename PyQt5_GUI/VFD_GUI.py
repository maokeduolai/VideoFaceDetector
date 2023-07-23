import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap, QIcon


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("Video Face Detector")

        # 设置窗口图标
        self.setWindowIcon(QIcon("images/VFD_Single_Logo.png"))  # 指定任务栏图标的文件路径

        # 创建导航区域
        navigation_widget = QWidget(self)
        navigation_layout = QVBoxLayout(navigation_widget)

        # 创建基本信息按钮
        basic_info_button = QPushButton(self)
        basic_info_button.setText("基本信息")
        basic_info_button.setIcon(QIcon("images/information.png"))  # 添加图标
        basic_info_button.clicked.connect(self.show_basic_info)
        basic_info_widget = QWidget(self)
        basic_info_layout = QVBoxLayout(basic_info_widget)
        basic_info_layout.addWidget(basic_info_button)
        basic_info_layout.addStretch(1)

        # 创建人物列表按钮
        person_list_button = QPushButton(self)
        person_list_button.setText("人物列表")
        person_list_button.setIcon(QIcon("images/face-detection.png"))  # 添加图标
        person_list_button.clicked.connect(self.show_person_list)
        person_list_widget = QWidget(self)
        person_list_layout = QVBoxLayout(person_list_widget)
        person_list_layout.addWidget(person_list_button)
        person_list_layout.addStretch(1)

        # 创建分类导出按钮
        export_button = QPushButton(self)
        export_button.setText("分类导出")
        export_button.setIcon(QIcon("images/export.png"))  # 添加图标
        export_button.clicked.connect(self.show_export)
        export_widget = QWidget(self)
        export_layout = QVBoxLayout(export_widget)
        export_layout.addWidget(export_button)
        export_layout.addStretch(1)

        # 将按钮添加到导航区域
        navigation_layout.addWidget(basic_info_widget)
        navigation_layout.addWidget(person_list_widget)
        navigation_layout.addWidget(export_widget)

        # 将导航区域添加到主窗口
        self.setCentralWidget(navigation_widget)

        # 创建主窗口的布局
        layout = QHBoxLayout()

        # 创建基本信息页面
        self.basic_info_widget = QWidget()
        basic_info_layout = QVBoxLayout()
        self.logo_label = QLabel()
        pixmap = QPixmap("images/VFD_Logo.png")
        scaled_pixmap = pixmap.scaled(305, 128)
        self.logo_label.setPixmap(scaled_pixmap)
        basic_info_layout.addWidget(self.logo_label)
        self.file_path_button = QPushButton("选择文件路径")
        self.file_path_button.clicked.connect(self.select_file_path)
        basic_info_layout.addWidget(self.file_path_button)
        self.file_path_label = QLabel("已选择的文件路径")
        basic_info_layout.addWidget(self.file_path_label)
        self.basic_info_widget.setLayout(basic_info_layout)

        # 创建人物列表页面
        self.person_list_widget = QWidget()
        person_list_layout = QVBoxLayout()
        self.person_list_textedit = QTextEdit()
        person_list_layout.addWidget(self.person_list_textedit)
        export_button = QPushButton("导出")
        export_button.clicked.connect(self.export_person_list)
        person_list_layout.addWidget(export_button)
        self.person_list_widget.setLayout(person_list_layout)

        # 创建分类导出页面
        self.export_widget = QWidget()
        export_layout = QVBoxLayout()
        self.export_textedit = QTextEdit()
        export_layout.addWidget(self.export_textedit)
        export_button = QPushButton("导出")
        export_button.clicked.connect(self.export_files)

        export_layout.addWidget(export_button)
        self.export_widget.setLayout(export_layout)

        # 将页面添加到主窗口布局中
        layout.addLayout(layout)
        layout.addWidget(self.basic_info_widget)
        layout.addWidget(self.person_list_widget)
        layout.addWidget(self.export_widget)

        # 创建主窗口的中心部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def create_nav_button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def show_basic_info(self):
        self.basic_info_widget.show()
        self.person_list_widget.hide()
        self.export_widget.hide()

    def show_person_list(self):
        self.basic_info_widget.hide()
        self.person_list_widget.show()
        self.export_widget.hide()

    def show_export(self):
        self.basic_info_widget.hide()
        self.person_list_widget.hide()
        self.export_widget.show()

    def select_file_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件路径")
        self.file_path_label.setText(file_path)

    def export_person_list(self):
        # 在这里添加导出人物列表的逻辑
        pass

    def export_files(self):
        # 在这里添加导出文件的逻辑
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())