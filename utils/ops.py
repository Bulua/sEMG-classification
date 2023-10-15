import contextlib
import time
import torch

class Profile(contextlib.ContextDecorator):

    def __init__(self, t=0.0) -> None:
        super().__init__()
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f'Elapsed time is {self.t} s'

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()