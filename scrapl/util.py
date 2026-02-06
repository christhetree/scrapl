import logging
import os
from typing import Dict, Tuple, Iterator

from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ReadOnlyTensorDict(nn.Module):
    def __init__(self, data: Dict[str | int, T], persistent: bool = True):
        super().__init__()
        self.persistent = persistent
        self.keys = set(data.keys())
        for k, v in data.items():
            self.register_buffer(f"tensor_{k}", v, persistent=persistent)

    def __getitem__(self, key: str | int) -> T:
        return self.get_buffer(f"tensor_{key}")

    def __contains__(self, key: str | int) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[str | int]:
        return iter(self.keys)

    def keys(self) -> Iterator[str | int]:
        return iter(self.keys)

    def values(self) -> Iterator[T]:
        for k in self.keys:
            yield self[k]

    def items(self) -> Iterator[Tuple[str | int, T]]:
        for k in self.keys:
            yield k, self[k]
