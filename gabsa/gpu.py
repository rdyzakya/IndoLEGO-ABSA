from pynvml import *
import torch


def print_gpu_utilization(index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU (index : {index}) memory occupied: {info.used//1024**2} MB.")

def print_all_gpu_utilization():
    for i in range(torch.cuda.device_count()):
        print_gpu_utilization(i)