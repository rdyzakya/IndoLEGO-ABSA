from pynvml import *


def print_gpu_utilization(index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU (index : {index}) memory occupied: {info.used//1024**2} MB.")

def print_all_gpu_utilization():
    i = 0
    while True:
        try:
            print_gpu_utilization(i)
            i += 1
        except:
            return