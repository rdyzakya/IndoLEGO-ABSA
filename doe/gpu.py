import subprocess

def get_gpu_memory_usage():
    """Returns a dictionary with GPU memory usage in MB for all available GPUs."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, universal_newlines=True)
    gpu_memory = [r.split(',') for r in result.stdout.strip().split('\n')]
    gpu_memory = [(int(used), int(total)) for used, total in gpu_memory]
    return {f"GPU {i}": {"used": used, "total": total} for i, (used, total) in enumerate(gpu_memory)}

def waiting_gpu_loop(n,threshold=5000):
    gpu_usage = get_gpu_memory_usage()
    while gpu_usage[f"GPU {n}"]["used"] > threshold:
        used = gpu_usage[f"GPU {n}"]["used"]
        total = gpu_usage[f"GPU {n}"]["total"]
        print(f"Waiting for GPU {n} is ready... | Used : {used} | Total : {total}",end="\r")
        if gpu_usage[f"GPU {n}"]["used"] <= threshold:
            print()
            break
        gpu_usage = get_gpu_memory_usage()