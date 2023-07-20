import sys

import dlib
import cv2
import os
import numpy as np
import ffmpeg


def convert_video(input_file, output_file):
    ffmpeg.input(input_file).output(output_file, r=target_fps, y='-y').run()


# 加载人脸检测器和人脸关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("source/model/shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("source/model/dlib_face_recognition_resnet_model_v1.dat")

# 加载原视频
video_path = "data/video/02_Test_Video.mp4"

# 创建保存人脸图像的目录
output_dir = "output/face"
os.makedirs(output_dir, exist_ok=True)

# 创建存储不同人脸特征向量的列表
face_descriptors = []

# 初始化帧计数器与人脸计数器
frame_count = 0
unique_face = 0

# 更改帧率并处理视频文件
target_fps = 5
temp_video_path = "output/temp/temp_video.mp4"
os.makedirs("output/temp", exist_ok=True)
convert_video(video_path, temp_video_path)

# 加载处理后的视频
cap = cv2.VideoCapture(temp_video_path)

while True:
    print(frame_count)

    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为灰度图像
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray_img)  # 上采样1次，对画面中较小人脸检测更友好，但计算时间也会增加。可调，例如:faces = detector(gray_img, 1)

    # 对每个检测到的人脸进行处理
    for face in faces:
        # 提取人脸关键点
        shape = predictor(gray_img, face)

        # 获取关键点坐标
        shape_points = []
        for n in range(68):
            x = shape.part(n).x
            y = shape.part(n).y
            shape_points.append((x, y))

        # 定义目标对齐点坐标
        desired_left_eye = (0.35, 0.35)
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

        aligned_face = aligned_face[y:y + h, x:x + w]

        output_path = os.path.join(output_dir, f"unique_face_{frame_count}.png")
        cv2.imwrite(output_path, aligned_face)

        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_descriptor = face_recognizer.compute_face_descriptor(RGB_img, shape, num_jitters=1)
        face_descriptor = np.array(face_descriptor)

        if len(face_descriptors) == 0:
            face_descriptors.append(face_descriptor)

        while len(face_descriptors) != 0:
            for i in range(0, len(face_descriptors)):
                distance = np.linalg.norm(face_descriptor - face_descriptors[i])
                if distance >= 0.4:
                    # 将人脸特征向量添加到列表中，同时记录人脸对应的视频
                    # 帧
                    face_descriptors.append(face_descriptor)
                    # # 保存帧
                    # output_path = os.path.join(output_dir, f"unique_face_{unique_face}.png")
                    # cv2.imwrite(output_path, frame)
                    unique_face += 1
                    break
            break

    frame_count += 1

# 释放视频文件
cap.release()

print("不同的人脸图像保存完成！")

# 退出程序
sys.exit()
