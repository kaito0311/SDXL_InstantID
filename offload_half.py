import torch
import functools
from typing import Dict, List, Mapping, Optional, Union

from accelerate.utils.operations import send_to_device, honor_type, is_torch_tensor
from accelerate.hooks import ModelHook, UserCpuOffloadHook, add_hook_to_module
from accelerate.utils.memory import clear_device_cache
from accelerate.state import PartialState


def send_to_device_half(tensor, device, non_blocking=False, skip_keys=None):

    if is_torch_tensor(tensor) or hasattr(tensor, "to"):
        # `torch.Tensor.to("npu")` could not find context when called for the first time (see this [issue](https://gitee.com/ascend/pytorch/issues/I8KECW?from=project-issue)).
        if device == "npu":
            device = "npu:0"
        if device == "xpu":
            device = "xpu:0"
        try:
            return tensor.to(device, non_blocking=non_blocking).half()
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.to(device).half()
        except AssertionError as error:
            raise error

    elif isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor,
            (
                send_to_device_half(
                    t, device, non_blocking=non_blocking, skip_keys=skip_keys
                )
                for t in tensor
            ),
        )
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                k: (
                    t
                    if k in skip_keys
                    else send_to_device_half(
                        t, device, non_blocking=non_blocking, skip_keys=skip_keys
                    )
                )
                for k, t in tensor.items()
            }
        )
    else:
        return tensor

def set_to_dtype(tensor, dtype, non_blocking=False, skip_keys=None):
    if is_torch_tensor(tensor) or hasattr(tensor, "type"):
        # `torch.Tensor.to("npu")` could not find context when called for the first time (see this [issue](https://gitee.com/ascend/pytorch/issues/I8KECW?from=project-issue)).
        try:
            return tensor.type(dtype)
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.type(dtype)
        except AssertionError as error:
            raise error

    elif isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor,
            (
                set_to_dtype(
                    t, dtype, non_blocking=non_blocking, skip_keys=skip_keys
                )
                for t in tensor
            ),
        )
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                k: (
                    t
                    if k in skip_keys
                    else set_to_dtype(
                        t, dtype, non_blocking=non_blocking, skip_keys=skip_keys
                    )
                )
                for k, t in tensor.items()
            }
        )
    else:
        return tensor

def get_dtype(data):
    """
    get dtype of data
    """
    if isinstance(data, Mapping):
        for obj in data.values():
            dtype_ = get_dtype(obj)
            if dtype_ is not None:
                return dtype_
    elif isinstance(data, (tuple, list)):
        for obj in data:
            dtype_ = get_dtype(obj)
            if dtype_ is not None:
                return dtype_
    elif isinstance(data, torch.Tensor):
        return data.dtype


class CpuOffLoadHalf(ModelHook):
    def __init__(
        self,
        execution_device: Optional[Union[str, int, torch.device]] = None,
        prev_module_hook: Optional["UserCpuOffloadHook"] = None,
    ):
        self.prev_module_hook = prev_module_hook

        self.execution_device = (
            execution_device
            if execution_device is not None
            else PartialState().default_device
        )
        self.input_dtype = None

    def init_hook(self, module):
        return module.to("cpu")

    def pre_forward(self, module, *args, **kwargs):
        if self.prev_module_hook is not None:
            self.prev_module_hook.offload()
            clear_device_cache()
        module.to(self.execution_device)

        if isinstance(self.execution_device, str) and "cuda" in self.execution_device:
            module = module.half()
        elif isinstance(self.execution_device, torch.device) and "cuda" in self.execution_device.type:
            module = module.half()
            


        if self.input_dtype is None:
            self.input_dtype = get_dtype([args, kwargs])

        return send_to_device_half(args, self.execution_device), send_to_device_half(
            kwargs, self.execution_device
        )

    def post_forward(self, module, output):

        self.init_hook(module)  # return to cpu

        output = set_to_dtype(output, self.input_dtype)

        self.input_dtype = None 

        return output


def half_cpu_offload_with_hook(model, execution_device=None, pre_module_hook=None):
    hook = CpuOffLoadHalf(
        execution_device=execution_device, prev_module_hook=pre_module_hook
    )
    add_hook_to_module(model, hook, append=True)
    user_hook = UserCpuOffloadHook(model, hook)

    return model, user_hook
