import tkinter
import os
import customtkinter
from PIL import Image

customtkinter.set_appearance_mode("Da")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Video Face Detector")
        self.geometry("1024x768")

        # 设置网格布局为 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # 加载图像
        image_path = "images"
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "VFD_Single_Logo.png")),
                                                 size=(52, 52))
        self.info_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "information.png")),
                                                 size=(26, 26))
        self.face_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "face-detection.png")),
                                                 size=(26, 26))
        self.export_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "export.png")),
                                                   size=(26, 26))
        self.large_logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "VFD_Logo.png")),
                                                       size=(305, 128))

        # 创建导航框架
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        # 创建Logo位置
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame,
                                                             text="  VFD",
                                                             image=self.logo_image,
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=30, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # 基本信息
        self.info_button = customtkinter.CTkButton(self.navigation_frame,
                                                   corner_radius=0,
                                                   height=40,
                                                   border_spacing=10,
                                                   text="基本信息",
                                                   fg_color="transparent",
                                                   text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   anchor="w",
                                                   image=self.info_image,
                                                   font=customtkinter.CTkFont(size=16))
        self.info_button.grid(row=1, column=0, sticky="ew")

        # 人物列表
        self.face_button = customtkinter.CTkButton(self.navigation_frame,
                                                   corner_radius=0,
                                                   height=40,
                                                   border_spacing=10,
                                                   text="人物列表",
                                                   fg_color="transparent",
                                                   text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   anchor="w",
                                                   image=self.face_image,
                                                   font=customtkinter.CTkFont(size=16))
        self.face_button.grid(row=2, column=0, sticky="ew")

        # 视频分类导出
        self.export_button = customtkinter.CTkButton(self.navigation_frame,
                                                     corner_radius=0,
                                                     height=40,
                                                     border_spacing=10,
                                                     text="分类导出",
                                                     fg_color="transparent",
                                                     text_color=("gray10", "gray90"),
                                                     hover_color=("gray70", "gray30"),
                                                     anchor="w",
                                                     image=self.export_image,
                                                     font=customtkinter.CTkFont(size=16))
        self.export_button.grid(row=3, column=0, sticky="ew")

        # 创建基本信息页框架
        self.info_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.info_frame.grid_columnconfigure(0, weight=1)

        # 基本信息页 Logo 图
        self.info_frame_large_logo_label = customtkinter.CTkLabel(self.info_frame,
                                                                  text="",
                                                                  image=self.large_logo_image)
        self.info_frame_large_logo_label.grid(row=2, column=1, padx=20, pady=10)


# command=self.home_button_event

if __name__ == "__main__":
    app = App()
    app.mainloop()
