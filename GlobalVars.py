import threading

# 定义全局变量
global_current_frame = 0
start_process = False
global_total_frame = 0

# 创建锁对象
lock = threading.Lock()
