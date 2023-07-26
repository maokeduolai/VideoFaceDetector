import os
import shutil
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QPixmap, QIcon, QTextTableFormat
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QFileDialog, QTextEdit
from qt_material import apply_stylesheet

from FaceAnalysisAndComparison import update_video_path, run_program, process_directory_input


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 自定义字体
        extra = {
            # Font
            'font_family': 'SimHei',
        }

        # 应用 QSS 样式
        apply_stylesheet(app, theme='dark_lightgreen.xml', invert_secondary=True, extra=extra)

        # 设置窗口标题
        self.setWindowTitle("Video Face Detector")

        # 设置窗口图标
        self.setWindowIcon(QIcon("images/VFD_Single_Logo.png"))

        # 创建主窗口的布局
        layout = QHBoxLayout()

        # 创建导航栏
        navigation_layout = QVBoxLayout()
        navigation_layout.addWidget(QLabel("   "))

        # 创建基本信息按钮
        basic_info_button = QPushButton("基本信息")
        basic_info_button.setIcon(QIcon("images/information.png"))
        basic_info_button.clicked.connect(self.show_basic_info)

        # 创建人物列表按钮
        self.person_list_button = QPushButton("人物列表")
        self.person_list_button.setIcon(QIcon("images/face-detection.png"))
        self.person_list_button.clicked.connect(self.show_person_list)

        # 创建分类导出按钮
        self.export_button = QPushButton("分类导出")
        self.export_button.setIcon(QIcon("images/export.png"))
        self.export_button.clicked.connect(self.show_export)

        # 设置按钮禁用条件
        os.makedirs("output/face", exist_ok=True)
        if len(os.listdir("output/face")) == 0:
            self.person_list_button.setEnabled(False)
            self.export_button.setEnabled(False)

        # 将按钮添加到导航栏中
        navigation_layout.addWidget(basic_info_button)
        navigation_layout.addWidget(self.person_list_button)
        navigation_layout.addWidget(self.export_button)

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
        self.folder_path_button = QPushButton("选择文件夹路径")
        self.folder_path_button.clicked.connect(self.select_folder_path)
        basic_info_layout.addWidget(self.folder_path_button)
        self.file_path_label = QLabel("已选择的文件/文件夹路径")
        basic_info_layout.addWidget(self.file_path_label)
        self.basic_info_widget.setLayout(basic_info_layout)

        # 创建人物列表页面
        self.person_list_widget = QWidget()
        person_list_layout = QVBoxLayout()
        self.person_list_textedit = QTextEdit()
        person_list_layout.addWidget(self.person_list_textedit)
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
        layout.addLayout(navigation_layout)
        layout.addWidget(self.basic_info_widget)
        layout.addWidget(self.person_list_widget)
        layout.addWidget(self.export_widget)

        # 创建主窗口的中心部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # 创建按钮函数
    def create_nav_button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    # 定义每个页面的显示内容
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

    # 载入临时文件夹中的所有图像
    def load_images(self):
        # 创建保存人脸图像的目录
        output_dir = "output/face"
        os.makedirs(output_dir, exist_ok=True)

        # 创建一个对象，用于操作 QTextEdit 的区域
        cursor = QTextCursor(self.person_list_textedit.document())

        # 设置并排图片的列数
        num_columns = 3

        # 计数器，用于确定何时插入新行
        counter = 0

        # 创建表格格式
        table_format = QTextTableFormat()
        table_format.setCellPadding(2)
        table_format.setCellSpacing(0)
        table_format.setBorder(0)

        table = cursor.insertTable(1, num_columns, table_format)

        # 显示每一张脸
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".png"):
                image_path = os.path.join(output_dir, file_name)
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaledToWidth(200)  # 调整图片宽度

                # 将 QPixmap 转换为 QImage
                image = scaled_pixmap.toImage()

                # 在当前单元格中插入图片
                cell_cursor = table.cellAt(0, counter).firstCursorPosition()
                cell_cursor.insertImage(image)

                # 在当前单元格中插入文件名
                cell_cursor.insertText("\n")
                cell_cursor.insertText(file_name)

                # 在文件名下方插入换行符
                cell_cursor.insertText("\n")

                # 更新计数器
                counter += 1

                # 插入新行
                if counter == num_columns:
                    table.appendRows(1)
                    counter = 0

    # 选择文件的方法
    def select_file_path(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件路径")

        if os.path.exists(file_path):
            self.file_path_label.setText(file_path)
            # 调用主程序的函数，将 file_path 传递给 video_path
            update_video_path(file_path)
            run_program()

            # 设置显示文本内容
            file_path = "output/face_list.txt"
            with open(file_path, "r") as file:
                content = file.read()
                self.export_textedit.setPlainText(content)

            self.person_list_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.load_images()

        QApplication.processEvents()

    # 选择文件夹的方法
    def select_folder_path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹路径")

        if os.path.exists(folder_path):
            self.file_path_label.setText(folder_path)
            update_video_path(folder_path)
            process_directory_input()

            # 设置显示文本内容
            file_path = "output/face_list.txt"
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    content = file.read()
                    self.export_textedit.setPlainText(content)
            else:
                print("文件不存在")

        self.person_list_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.load_images()

        QApplication.processEvents()

    # 导出人物与视频文件关联 txt 文件
    def export_files(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹路径")

        # 复制出文件
        source_file = "output/face_list.txt"
        destination_folder = folder_path
        shutil.copy(source_file, destination_folder)


if __name__ == "__main__":
    export_folder_path = 'output'
    # 初始化临时存储文件夹
    if os.path.exists(export_folder_path):
        # 删除文件夹及其内容
        shutil.rmtree(export_folder_path)
    else:
        print("文件夹不存在，无需删除。")

    # 设置根据屏幕DPI缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
