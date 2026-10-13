import os
import sys

# 修复 Windows 上 torch 2.11 + Python 3.13 的两个已知问题：
# 1) CUDA DLL（caffe2_nvrtc.dll）的依赖找不到 -> 手动把 torch/lib 加入 DLL 搜索路径
# 2) numpy(MKL) 与 torch 的 OpenMP 冲突（libiomp5md.dll 重复加载）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if sys.platform == "win32":
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.submodule_search_locations:
            torch_lib = os.path.join(spec.submodule_search_locations[0], "lib")
            if os.path.isdir(torch_lib):
                os.add_dll_directory(torch_lib)
    except Exception:
        pass
