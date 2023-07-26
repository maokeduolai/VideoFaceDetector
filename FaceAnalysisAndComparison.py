import glob
import os

import cv2
import dlib
import ffmpeg
import numpy as np
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QProgressBar, QApplication, QWidget, QVBoxLayout

import GlobalVars

video_path = ""


# 定义函数，用于更新 video_path 的值
def update_video_path(new_path):
    global video_path
    video_path = new_path


def process_directory_input():
    # 选取所有的视频文件
    video_files = glob.glob(os.path.join(video_path, '*.mp4')) + glob.glob(
        os.path.join(video_path, '*.avi')) + glob.glob(os.path.join(video_path, '*.mov')) + glob.glob(
        os.path.join(video_path, '*.wmv'))

    # 遍历所有视频文件，并更新处理的视频文件路径
    for video_file in video_files:
        update_video_path(video_file)
        run_program()


class ProgressBarWindow(QWidget):
    def __init__(self, total_frame):
        super().__init__()
        self.setWindowTitle("正在处理中...")

        # 设置窗口图标
        self.setWindowIcon(QIcon("images/VFD_Single_Logo.png"))

        self.layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, total_frame)
        self.progress_bar.setFormat("%p%")  # 设置进度条显示格式为百分比
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

        # 设置窗口大小
        self.setFixedSize(600, 100)

    def set_value(self, current_frame):
        self.progress_bar.setValue(current_frame)


# 创建存储不同人脸特征向量的列表
face_descriptors = []

# 创建人脸与视频对应列表
face_list = []

# 人脸连续计数
unique_face = 1


def run_program():
    global face_list
    global unique_face

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

    # 获取视频文件名
    video_filename = video_path.split("/")[-1]

    # 显示进度条
    progress_window = ProgressBarWindow(GlobalVars.global_total_frame)
    progress_window.show()

    # 加载处理后的视频
    cap = cv2.VideoCapture(temp_video_path)

    while True:
        print(frame_count)

        # 获取锁对象
        lock = GlobalVars.lock
        with lock:
            GlobalVars.global_current_frame = frame_count

        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

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

            # 定义目标尺寸
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

            # 截取对齐后的人脸
            aligned_face = aligned_face[y:y + h, x:x + w]

            # 将OpenCV默认存储的 BGR 通道顺序转换为 RGB
            RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 得到人脸的 128D 数据
            # 较小的 num_jitters 值可以提供较快的计算速度，但可能会降低准确性；而较大的 num_jitters 值可以提供更准确的结果，但计算时间会增加
            face_descriptor = np.array(face_recognizer.compute_face_descriptor(RGB_img, shape, num_jitters=2))

            output_dir = "output/face"

            # 存储第一张人脸的数据，输出人脸图片，然后对比下一张人脸
            if len(face_descriptors) == 0:
                face_descriptors.append(face_descriptor)
                output_path = os.path.join(output_dir, f"Person_{unique_face}.png")
                cv2.imwrite(output_path, aligned_face)

                # 添加人脸 ID 到数组
                face_list.append(f"Person_{unique_face} -> {video_filename}")

                print(unique_face)
                print(video_filename)

                unique_face += 1
                continue

            # 测试是否是相同人脸
            is_unique = True
            for i in range(0, len(face_descriptors)):
                distance = np.linalg.norm(face_descriptor - face_descriptors[i])
                print(distance)  # 输出欧氏距离，便于调试

                # 欧式距离小于0.6，判断为相同人脸
                if distance < 0.6:
                    is_unique = False
                    break

            # 对不同的人脸做处理
            if is_unique:
                # 将人脸特征向量添加到列表中
                face_descriptors.append(face_descriptor)
                # 保存帧
                output_path = os.path.join(output_dir, f"Person_{unique_face}.png")
                cv2.imwrite(output_path, aligned_face)

                # 添加人脸 ID 到数组
                face_list.append(f"Person_{unique_face} -> {video_filename}")

                print(unique_face)
                print(video_filename)

                unique_face += 1

            # 更新进度条
            progress_window.set_value(GlobalVars.global_current_frame)
            QApplication.processEvents()

        frame_count += 1

    # 释放视频文件
    cap.release()

    # 向 txt 文件写入人脸与视频关联
    file_path = "output/face_list.txt"

    try:
        # 打开文件并写入内容
        with open(file_path, "a") as file:
            for item in face_list:
                file.write(item + "\n")
        print(f"文件 '{file_path}' 创建成功并写入内容！")
    except Exception as e:
        print(f"创建文件 '{file_path}' 失败：{str(e)}")

    print(face_descriptors)

    print("不同的人脸图像保存完成！")
