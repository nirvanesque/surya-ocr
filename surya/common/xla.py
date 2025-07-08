from surya.settings import settings

if settings.TORCH_DEVICE_MODEL == "xla":
    import torch_xla.core.xla_model as xm
else:
    xm = None


def mark_step():
    if xm is not None:
        xm.mark_step()


def get_compile_args(device: str) -> dict:
    if device != "xla":
        return {}

    return {
        "backend": "openxla",
    }
