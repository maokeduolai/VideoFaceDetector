import dlib
import cv2
import os
import numpy as np

# 加载人脸检测器和人脸关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("source/shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("source/dlib_face_recognition_resnet_model_v1.dat")

# 加载视频文件
video_path = "source/Test_Video.mp4"
cap = cv2.VideoCapture(video_path)

# 创建保存人脸图像的目录
output_dir = "output/face"
os.makedirs(output_dir, exist_ok=True)

# 创建存储人脸特征向量的列表
face_descriptors = []

# 读取视频帧并提取人脸
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 将帧转换为灰度图像
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray_img, 1)  # 上采样1次，对画面中多人脸检测更友好

    # 对每个检测到的人脸进行处理
    for face in faces:
        # 提取人脸关键点
        shape = predictor(gray_img, face)
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_descriptor = face_recognizer.compute_face_descriptor(RGB_img, shape, num_jitters=10)
        face_descriptor = np.array(face_descriptor)

        # 将人脸特征向量添加到列表中，同时记录人脸对应的视频帧
        face_descriptors.append([face_descriptor, frame_count])

    frame_count += 1

# 比较人脸数据的差异
unique_faces = []

for i, face_descriptor in enumerate(face_descriptors):
    is_unique = True
    for j in range(i + 1, len(face_descriptors)):
        distance = np.linalg.norm(face_descriptor[0] - face_descriptors[j][0])
        if distance < 0.3:
            is_unique = False
            break
    if is_unique:
        unique_faces.append([face_descriptor, face_descriptor[1]])

# 保存不同的人脸图像
for i, array in enumerate(unique_faces):
    cap = cv2.VideoCapture(video_path)  # 重新打开视频以便保存
    face_descriptor = array[0]

    # 得到需要保存的帧位置
    frame_to_save = array[1]

    # 循环保存指定帧
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count == frame_to_save:
            # 保存帧
            output_path = os.path.join(output_dir, f"unique_face_{i}.png")
            cv2.imwrite(output_path, frame)
            break

# 释放视频文件
cap.release()

print("不同的人脸图像保存完成！")
