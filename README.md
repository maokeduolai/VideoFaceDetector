# 一个基于人脸的视频分类器

## 功能

该程序是一个使用dlib库进行人脸检测和人脸识别的示例程序。它可以读取视频文件，检测视频中的人脸，并保存不同人脸的图像。

- 从一个或一组视频中导出该视频内出现的所有人脸

- 选取一个或多个人脸显示对应视频列表

- 选取一个或多个人脸自动给视频分类

## 安装和运行

**注意：本程序适合在有 NVIDIA GPU 的计算机上运行**

- Python版本要求：3.10.12

- GPU最低支持CUDA版本：11.6.2（查询显卡文档确认）

- 需安装Anaconda最新版本

## 命令行安装

1. 确保已安装Anaconda环境。如果未安装，请根据官方文档进行安装：[Anaconda官方文档](https://docs.anaconda.com/)

2. 克隆或下载程序代码。

3. 在命令行或终端中，进入程序代码所在的目录。

4. 使用以下命令创建并激活程序所需的环境：

   ```shell
   conda env create -f environment.yml
   conda activate <environment_name>
   ```

   其中，`<environment_name>` 是你想要为环境指定的名称。

5. 安装FFmpeg。你可以使用以下命令安装FFmpeg：

   ```shell
   conda install -c conda-forge ffmpeg
   ```

## 运行

1. 在命令行或终端中，进入程序代码所在的目录。

2. 使用以下命令激活程序所需的环境：

   ```shell
   conda activate <environment_name>
   ```

   其中，`<environment_name>` 是你在安装步骤中为环境指定的名称。

3. 运行程序：

   ```shell
   python FaceAnalysisAndComparison.py
   ```

4. 程序开始运行后，它会读取指定的视频文件并进行人脸检测和人脸识别。程序会在输出目录中保存不同人脸的图像。

5. 程序运行结束后，你可以在输出目录中找到保存的人脸图像。

## 注意事项

- 确保已在程序中正确设置视频文件的路径和其他参数。
- 确保已提供正确的人脸关键点检测器和人脸识别模型文件。
- 程序使用dlib库进行人脸检测和人脸识别，因此需要安装dlib库和其依赖项。

## 依赖库

- dlib（支持CUDA的版本暂时还需要编译）
- opencv-python
- numpy
- ffmpeg

## 资源

- [dlib官方文档](http://dlib.net/)
- [OpenCV官方文档](https://docs.opencv.org/)
- [NumPy官方文档](https://numpy.org/doc/)
- [FFmpeg官方文档](https://ffmpeg.org/documentation.html)
- 所有的图标来源于：<a href="https://www.flaticon.com" title="export icons">Icons created by
  Freepik</a>