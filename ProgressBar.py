import sys
import time

from PyQt5.QtCore import QRect, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QDialog, QProgressBar

import GlobalVars


class ProgressBar(QDialog):
    def __init__(self, parent=None):
        super(ProgressBar, self).__init__(parent)

        # QDialog 窗体的设置
        self.resize(500, 32)

        # 创建并设置 QProcessbar
        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(GlobalVars.global_total_frame)
        self.progressBar.setValue(0)
        self.progressBar.setGeometry(QRect(1, 3, 499, 28))  # 设置进度条在 QDialog 中的位置 [左，上，右，下]
        self.show()

    def set_value(self, task_number, total_task_number, value):
        if task_number == '0' and total_task_number == '0':
            self.setWindowTitle(self.tr('正在处理中'))
        else:
            label = "正在处理：" + "第" + str(task_number) + "/" + str(total_task_number) + '帧'
            self.setWindowTitle(self.tr(label))
        self.progressBar.setValue(value)


class PyQtBar:

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.progressbar = ProgressBar()

    task_id = 1
    total_frame = GlobalVars.global_total_frame

    def set_value(self, task_number, total_task_number, i):
        self.progressbar.set_value(str(task_number), str(total_task_number), i + 1)  # 更新进度条的值
        QApplication.processEvents()  # 实时刷新显示

    def close(self):
        self.progressbar.close()  # 关闭进度条
        self.app.exit()  # 关闭系统 app


def run_program():
    # 使用示例
    bar = PyQtBar()  # 创建实例
    total_frame = GlobalVars.global_total_frame

    task_id = 1  # 子任务序号
    while True:
        time.sleep(0.05)
        current_frame = GlobalVars.global_current_frame
        bar.set_value(task_id, total_frame, current_frame)  # 刷新进度条

        if current_frame == total_frame:
            bar.close()  # 关闭 bar 和 app

        def do_work(callback):
            callback()
