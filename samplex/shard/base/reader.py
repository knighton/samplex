from typing import Any

from samplex.util.schemata import Schemata


class ShardReader:
    def __init__(
        self,
        schemata: Schemata,
        num_samples: int,
    ):
        self.schemata = schemata
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getoneitem__(self, sample_id: int) -> dict[str, Any]:
        raise NotImplementedError
