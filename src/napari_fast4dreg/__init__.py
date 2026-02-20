__version__ = "0.0.1"

from ._fast4Dreg_functions import get_gpu_info, set_gpu_acceleration
from ._widget import Fast4DReg_widget
from .api import fast4dreg, register, register_image, register_image_from_file

__all__ = (
    "Fast4DReg_widget",
    "register_image",
    "register_image_from_file",
    "fast4dreg",
    "register",
    "set_gpu_acceleration",
    "get_gpu_info",
)
