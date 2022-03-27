import torch


class CBuffer:
    def __init__(self, buffer_size):
        self.buffer = torch.FloatTensor(buffer_size, 2).uniform_(-1.0, 1.0)

        self.buffer_size = buffer_size
        self.ptr = 0
        self.pushed_num = 0

    def push(self, samples):
        n = len(samples)

        if self.ptr + n >= self.buffer_size:
            reminder = self.buffer_size - self.ptr
            self.buffer[-reminder:] = samples[:reminder]
            self.buffer[:n - reminder] = samples[reminder:]
        else:
            self.buffer[self.ptr:self.ptr + n] = samples

        self.ptr = (self.ptr + n) % self.buffer_size
        self.pushed_num = min(self.pushed_num + 1, self.buffer_size)

    def sample(self, n, limit_size=True):
        if limit_size:
            idxs = torch.randint(0, self.pushed_num, size=(n,))
        else:
            idxs = torch.randint(0, self.buffer_size, size=(n,))

        return self.buffer[idxs]
