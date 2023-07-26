# 基于人脸的视频分类器

## 功能

该程序是一个使用 dlib 库进行人脸检测和人脸识别的程序。它可以读取视频文件，检测视频中的人脸，并显示不同人脸的图像

- 显示一个或一组视频中出现的所有人脸

- 导出人脸与视频的对应列表

## 安装和运行

**注意：本程序适合在有 NVIDIA GPU 的计算机上运行**

- Python版本要求：3.10.12

- GPU最低支持CUDA版本：11.6.2（查询显卡文档确认）

- 需安装Anaconda最新版本

### 1.安装

1. 确保已安装 Anaconda 环境。如果未安装，请根据官方文档进行安装：[Anaconda官方文档](https://docs.anaconda.com/)

2. 克隆或下载程序代码

3. 在命令行或终端中，进入程序代码所在的目录

4. 使用以下命令创建并激活程序所需的环境：

   ```shell
   conda env create -f environment.yml
   conda activate VideoFaceDetector
   ```
   
5. 安装 dlib 的两种方式
   1. 根据以下教程，安装支持 CUDA 的 dlib 库

      [基于 Anaconda 的 dlib（CUDA 版本）安装的终极解决方案 - Arisa _ Blog](reference/基于%20Anaconda%20的%20dlib（CUDA%20版本）安装的终极解决方案%20-%20Arisa%20_%20Blog.html)

   2. 使用 Anaconda 安装仅支持 CPU 运算的 dlib 库（计算速度极慢）
      
      ```shell
      conda install -c conda-forge dlib
      ```

### 2.运行

1. 在命令行或终端中，进入程序代码所在的目录

2. 使用以下命令激活程序所需的环境：

   ```shell
   conda activate VideoFaceDetector
   ```

3. 运行程序：

   ```shell
   python VFD_GUI.py
   ```

4. 程序开始运行后，点击 **选择文件路径** 或 **选择文件夹路径** 选择需要处理的视频文件或目录

5. 稍等一会儿后，程序开始对视频的处理，弹出 **正在处理中...** 窗口

6. 处理完后，界面的 **人物列表** 与 **分类导出** 按钮可用，点击他们可以进入各自的页面

7. **人物列表** 页面显示视频中出现的所有人物，**分类导出** 页面显示人脸与视频的对应关系

8. 点击 **分类导出** 页面中的 **导出** 按钮，选择导出文件夹，在文件夹内会生成包含对应列表的 txt 文本文件

## 注意事项

- 确保人脸关键点检测器和人脸识别模型文件在默认位置

## 依赖库

### 1.通过 Anaconda 安装
- bzip2 = 1.0.8
- ca-certificates = 2023.7.22
- cuda = 11.6.2
- cuda-cccl = 12.2.53
- cuda-command-line-tools = 12.2.0
- cuda-compiler = 12.2.0
- cuda-cudart = 12.2.53
- cuda-cudart-dev = 12.2.53
- cuda-cuobjdump = 12.2.53
- cuda-cupti = 12.2.60
- cuda-cuxxfilt = 12.2.53
- cuda-documentation = 12.2.53
- cuda-libraries = 12.2.0
- cuda-libraries-dev = 12.2.0
- cuda-nsight-compute = 12.2.0
- cuda-nvcc = 12.2.91
- cuda-nvdisasm = 12.2.53
- cuda-nvml-dev = 12.2.81
- cuda-nvprof = 12.2.60
- cuda-nvprune = 12.2.53
- cuda-nvrtc = 12.2.91
- cuda-nvrtc-dev = 12.2.91
- cuda-nvtx = 12.2.53
- cuda-nvvp = 12.2.60
- cuda-opencl = 12.2.53
- cuda-opencl-dev = 12.2.53
- cuda-profiler-api = 12.2.53
- cuda-runtime = 12.2.0
- cuda-sanitizer-api = 12.2.53
- cuda-toolkit = 12.2.0
- cuda-tools = 12.2.0
- cuda-visual-tools = 12.2.0
- cudatoolkit = 11.3.1
- cudnn = 8.2.1
- eigen = 3.4.0
- ffmpeg = 4.2.3
- freetype = 2.10.4
- glib = 2.69.1
- gst-plugins-base = 1.18.5
- gstreamer = 1.18.5
- hdf5 = 1.10.6
- icu = 58.2
- intel-openmp = 2023.1.0
- jpeg = 9e
- lerc = 3.0
- libblas = 3.9.0
- libcblas = 3.9.0
- libclang = 12.0.0
- libcublas = 12.2.1.16
- libcublas-dev = 12.2.1.16
- libcufft = 11.0.8.15
- libcufft-dev = 11.0.8.15
- libcurand = 10.3.3.53
- libcurand-dev = 10.3.3.53
- libcusolver = 11.5.0.53
- libcusolver-dev = 11.5.0.53
- libcusparse = 12.1.1.53
- libcusparse-dev = 12.1.1.53
- libdeflate = 1.17
- libffi = 3.4.4
- libiconv = 1.17
- liblapack = 3.9.0
- libnpp = 12.1.1.14
- libnpp-dev = 12.1.1.14
- libnvjitlink = 12.2.91
- libnvjitlink-dev = 12.2.91
- libnvjpeg = 12.2.0.2
- libnvjpeg-dev = 12.2.0.2
- libogg = 1.3.4
- libpng = 1.6.39
- libprotobuf = 3.20.3
- libtiff = 4.5.0
- libvorbis = 1.3.7
- libwebp = 1.2.4
- libwebp-base = 1.2.4
- libxml2 = 2.10.3
- libxslt = 1.1.37
- lz4-c = 1.9.4
- mkl = 2020.4
- nsight-compute = 2023.2.0.16
- numpy = 1.22.3
- opencv = 4.6.0
- openssl = 1.1.1u
- packaging = 23.0
- pcre = 8.45
- pillow = 9.4.0
- pip = 23.1.2
- ply = 3.11
- pyqt5-sip = 12.11.0
- python = 3.10.12
- python_abi = 3.10
- qt-main = 5.15.2
- qt-webengine = 5.15.9
- qtwebkit = 5.212
- setuptools = 67.8.0
- sip = 6.6.2
- sqlite = 3.41.2
- tk = 8.6.12
- toml = 0.10.2
- tzdata = 2023c
- vc = 14.2
- vs2015_runtime = 14.27.29016
- wheel = 0.38.4
- xz = 5.4.2
- zlib = 1.2.13
- zstd = 1.5.5

### 2.通过 pip 安装
- click == 8.1.6
- colorama == 0.4.6
- customtkinter == 5.2.0
- darkdetect == 0.7.1
- ffmpeg-python == 0.2.0
- future == 0.18.3
- jinja2 == 3.1.2
- markupsafe == 2.1.3
- pyqt5 == 5.15.9
- pyqt5-plugins == 5.15.9.2.3
- pyqt5-qt5 == 5.15.2
- pyqt5-tools == 5.15.9.3.3
- python-dotenv == 1.0.0
- qt-material == 2.14
- qt5-applications == 5.15.2.2.3
- qt5-tools == 5.15.2.1.3

## 资源

- [dlib 官方文档](http://dlib.net/)
- [CUDA 官方文档](https://docs.nvidia.com/cuda/)
- [FFmpeg 官方文档](https://ffmpeg.org/documentation.html)
- [Numpy 官方文档](https://numpy.org/doc/)
- [opencv 官方文档](https://docs.opencv.org/)
- [PyQt5 官方文档](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [Python 官方文档](https://docs.python.org/)
- 软件中图标来源于：**[Flaticon](https://www.flaticon.com)**
- 测试视频来源于：**[Pixabay](https://pixabay.com/)**