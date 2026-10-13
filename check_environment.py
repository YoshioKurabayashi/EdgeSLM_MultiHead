"""Read-only environment probe; does not load or train a model."""
import importlib.metadata
import json
import platform

import torch


def main():
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "scikit-learn", "pytest", "psutil")},
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        report.update({"gpu": torch.cuda.get_device_name(0),
                       "compute_capability": list(torch.cuda.get_device_capability(0)),
                       "vram_bytes": torch.cuda.get_device_properties(0).total_memory})
        value = torch.ones(16, device="cuda").sum().item()
        report["cuda_tensor_check"] = value == 16.0
        if value != 16.0:
            raise AssertionError("GPU tensor operation failed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
