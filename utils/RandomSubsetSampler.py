from torch.utils.data.sampler import Sampler
from typing import Sized, Sequence, Iterator
import torch
import numpy as np

class RandomSubsetSampler(Sampler[int]):
    def __init__(self, num_samples: int, indice: Sequence[int], generator=None):
        self._num_samples = num_samples
        self.indice = np.array(indice)
        self.generator = generator
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.indice)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        
        for _ in range(self._num_samples // 32):
            yield from self.indice[torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()].tolist()
        yield from self.indice[torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()].tolist()

    def __len__(self) -> int:
        return self._num_samples