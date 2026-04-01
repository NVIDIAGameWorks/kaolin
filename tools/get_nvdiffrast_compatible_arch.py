import subprocess

from torch.utils.cpp_extension import CUDA_HOME

def get_cuda_bare_metal_version(cuda_dir):
    """Get CUDA version from nvcc."""
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    return raw_output, release[0], release[1][0]

if __name__ == "__main__":
    if CUDA_HOME is None:
        raise RuntimeError("Pytorch doesn't have CUDA_HOME set.")
    _, major, minor = get_cuda_bare_metal_version(CUDA_HOME)
    major, minor = int(major), int(minor)
    if major == 11:
        if minor == 0:
            arch_list = "6.0;6.1;6.2;7.0;7.5;8.0"
        elif minor < 8:
            arch_list = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        else:
            arch_list = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9"
    elif major == 12:
        if minor <= 6:
            arch_list = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif minor == 8:
            arch_list = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;12.0"
        else:
            arch_list = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;12.0;12.1"
    elif major == 13:
        arch_list = "7.5;8.0;8.6;8.9;9.0;12.0;12.1"
    else:
        arch_list = "6.0;6.1;6.2;7.0;7.5"
    print(arch_list)

