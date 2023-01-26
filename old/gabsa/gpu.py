from pynvml import *


def print_gpu_utilization(index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

def print_all_gpu_utilization():
    i = 0
    while True:
        try:
            used = print_gpu_utilization(i)
            print(f"GPU (index : {i}) memory occupied: {used//1024**2} MB.")
            i += 1
        except:
            return