import contextlib
import torch
import funcy


class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if (
    torch.cuda.is_available()
    and hasattr(torch.cuda, "amp")
    and hasattr(torch.cuda.amp, "autocast")
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:

    @contextlib.contextmanager
    def autocast():
        yield


@contextlib.contextmanager
def inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield
