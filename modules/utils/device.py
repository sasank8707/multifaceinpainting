"""
Hardware detection, device dispatch, and memory management module.
Ensures seamless execution across CPU (smoke tests / local verification)
and CUDA GPU (8-12 GB VRAM training and evaluation clusters).
"""

import os
import platform
import psutil
from typing import Dict, Any, Tuple


def get_hardware_profile() -> Dict[str, Any]:
    """Inspects and returns a dictionary of host system hardware resources."""
    profile = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": {
            "name": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "gpu": {
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }
    }
    
    try:
        import torch
        profile["gpu"]["cuda_available"] = torch.cuda.is_available()
        profile["gpu"]["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                profile["gpu"]["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                })
    except ImportError:
        profile["gpu"]["error"] = "PyTorch not yet imported."

    return profile


from dataclasses import dataclass


@dataclass
class DeviceConfig:
    """Encapsulates execution device and numerical precision settings."""
    device_str: str
    dtype: Any

    @property
    def device(self):
        import torch
        return torch.device(self.device_str)

    @property
    def is_cuda(self) -> bool:
        return "cuda" in self.device_str


def get_optimal_device(preferred: str = "auto") -> Tuple[str, Any]:
    """
    Returns optimal device string ('cuda', 'cuda:0', or 'cpu') and torch dtype.
    Handles fallbacks gracefully.
    """
    import torch
    
    if preferred == "cuda" and torch.cuda.is_available():
        device_str = "cuda:0"
        dtype = torch.float16
    elif preferred == "cpu":
        device_str = "cpu"
        dtype = torch.float32
    else: # "auto"
        if torch.cuda.is_available():
            device_str = "cuda:0"
            dtype = torch.float16
        else:
            device_str = "cpu"
            dtype = torch.float32

    return device_str, dtype


def get_device_config(preferred: str = "auto") -> DeviceConfig:
    """Returns DeviceConfig instance wrapping get_optimal_device output."""
    dev_str, dt = get_optimal_device(preferred=preferred)
    return DeviceConfig(device_str=dev_str, dtype=dt)


def print_hardware_summary():
    """Prints formatted hardware summary for research reproducibility logs."""
    profile = get_hardware_profile()
    print("=" * 60)
    print("HARDWARE & EXECUTION ENVIRONMENT PROFILE")
    print("=" * 60)
    print(f"OS Platform     : {profile['platform']}")
    print(f"Python Version  : {profile['python_version']}")
    print(f"CPU             : {profile['cpu']['name']} ({profile['cpu']['physical_cores']} cores, {profile['cpu']['logical_cores']} threads)")
    print(f"System RAM      : {profile['ram']['total_gb']} GB total ({profile['ram']['available_gb']} GB free)")
    print(f"CUDA Available  : {profile['gpu']['cuda_available']}")
    if profile['gpu']['cuda_available']:
        for dev in profile['gpu']['devices']:
            print(f"GPU [{dev['index']}]         : {dev['name']} ({dev['total_vram_gb']} GB VRAM)")
    else:
        print("GPU             : Integrated / CPU execution mode active.")
    print("=" * 60)


if __name__ == "__main__":
    print_hardware_summary()
