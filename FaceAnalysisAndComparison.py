import glob
import os
import sys
import time

import cv2
import dlib
import ffmpeg
import numpy as np
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QProgressBar, QDialog, QApplication

import GlobalVars

# import ProgressBar

video_path = ""


# 定义函数，用于更新 video_path 的值
def update_video_path(new_path):
    global video_path
    video_path = new_path


# 创建存储不同人脸特征向量的列表
face_descriptors = []


def process_directory_input():
    if os.path.isdir(video_path):
        video_files = glob.glob(os.path.join(video_path, '*.mp4')) + glob.glob(
            os.path.join(video_path, '*.avi')) + glob.glob(os.path.join(video_path, '*.mov')) + glob.glob(
            os.path.join(video_path, '*.wmv'))
        # 遍历所有视频文件，并更新处理的视频文件路径
        if video_path:
            for video_file in video_files:
                update_video_path(video_file)
                run_program()


def run_program():
    # 获取文件绝对路径，加载模型
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shape_predictor_path = os.path.join(current_dir, "source/model/shape_predictor_68_face_landmarks.dat")
    face_recognizer_path = os.path.join(current_dir, "source/model/dlib_face_recognition_resnet_model_v1.dat")

    # 加载人脸检测器和人脸关键点检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)

    # 初始化帧计数器与人脸计数器
    frame_count = 0
    unique_face = 1

    # 更改帧率并处理视频文件
    def convert_video(input_file, output_file):
        ffmpeg.input(input_file).output(output_file, r=target_fps, y='-y').run()

    target_fps = 5
    temp_video_path = "output/temp/temp_video.mp4"
    os.makedirs("output/temp", exist_ok=True)
    convert_video(video_path, temp_video_path)

    # 读取处理后的视频帧数
    probe = ffmpeg.probe(temp_video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    GlobalVars.global_total_frame = int(video_info['nb_frames'])

    # 加载处理后的视频
    cap = cv2.VideoCapture(temp_video_path)

    keep_running = True

    while keep_running:
        print(frame_count)

        # 获取锁对象
        lock = GlobalVars.lock
        with lock:
            GlobalVars.global_current_frame = frame_count

        ProgressBar.run_program()

        def continue_execution():
            global keep_running
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                keep_running = False

            # 检测人脸
            faces = detector(frame)  # 上采样1次，对画面中较小人脸检测更友好，但计算时间也会增加。可调，例如:faces = detector(gray_img, 1)

            # 对每个检测到的人脸进行处理
            for face in faces:
                # 提取人脸关键点
                shape = predictor(frame, face)

                # 获取关键点坐标
                shape_points = []
                for n in range(68):
                    x = shape.part(n).x
                    y = shape.part(n).y
                    shape_points.append((x, y))

                # 定义目标对齐点坐标
                # desired_left_eye = (0.35, 0.35)
                desired_face_width = 1024
                desired_face_height = 1024

                # 计算旋转和缩放参数
                eyes_center = ((shape_points[36][0] + shape_points[45][0]) // 2,
                               (shape_points[36][1] + shape_points[45][1]) // 2)
                angle = np.degrees(np.arctan2(shape_points[45][1] - shape_points[36][1],
                                              shape_points[45][0] - shape_points[36][0])) * 1.0  # 乘1.0转换数据类型为float
                scale = np.sqrt((desired_face_width ** 2 + desired_face_height ** 2) / (
                        (shape_points[45][0] - shape_points[36][0]) ** 2 + (
                        shape_points[45][1] - shape_points[36][1]) ** 2)) * 0.3

                # 构建仿射变换矩阵
                M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

                # 进行仿射变换
                aligned_face = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)

                # 截取对齐后的人脸区域
                x = int(eyes_center[0] - desired_face_width / 2)
                y = int(eyes_center[1] - desired_face_height / 2)
                w = desired_face_width
                h = desired_face_height

                # 获取帧的宽度和高度
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]

                # 进行边界检查
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

                # 截取
                aligned_face = aligned_face[y:y + h, x:x + w]

                # 将OpenCV默认存储的 BGR 通道顺序转换为 RGB
                RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 得到人脸的 128D 数据
                # 较小的 num_jitters 值可以提供较快的计算速度，但可能会降低准确性；而较大的 num_jitters 值可以提供更准确的结果，但计算时间会增加
                face_descriptor = np.array(face_recognizer.compute_face_descriptor(RGB_img, shape, num_jitters=3))

                # 创建保存人脸图像的目录
                output_dir = "output/face"

                global unique_face

                # 存储第一张人脸的数据，输出人脸图片，然后对比下一张人脸
                if len(face_descriptors) == 0:
                    face_descriptors.append(face_descriptor)
                    output_path = os.path.join(output_dir, f"unique_face_{unique_face}.png")
                    cv2.imwrite(output_path, aligned_face)
                    unique_face += 1
                    continue

                # 测试是否有相同人脸
                is_unique = True
                for i in range(0, len(face_descriptors)):
                    distance = np.linalg.norm(face_descriptor - face_descriptors[i])
                    print(distance)
                    if distance < 0.6:
                        is_unique = False
                        break

                if is_unique:
                    # 将人脸特征向量添加到列表中
                    face_descriptors.append(face_descriptor)
                    # 保存帧
                    output_path = os.path.join(output_dir, f"unique_face_{unique_face}.png")
                    cv2.imwrite(output_path, aligned_face)
                    unique_face += 1

                def append_to_file(file_path, content):
                    with open(file_path, 'a') as file:
                        file.write(content)

                # 示例内容，你可以根据需要修改
                additional_info = "Occupation: Software Engineer"

                # 调用函数追加内容到文件
                append_to_file("output/face.txt", additional_info)

            global frame_count
            frame_count += 1

        do_work = ProgressBar.run_program()
        do_work(continue_execution)

    # 释放视频文件
    cap.release()

    print(face_descriptors)

    print("不同的人脸图像保存完成！")

    return 0


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
