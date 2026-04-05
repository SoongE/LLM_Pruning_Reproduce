# dataset_safetensors.py
import bisect
import json
from typing import Any, Dict, Tuple

from safetensors import safe_open
from torch.utils.data import Dataset


class SafeTensorShards(Dataset):
    """
    Dataset over many .safetensors shards, each with a batched tensor:
      - feature_key: [N_i, ...]
      - label_key  : [N_i, ...] (optional)
    Uses a manifest.json produced by build_manifest.py
    """

    def __init__(self, manifest_path: str, return_meta: bool = False, sticky_handles: bool = True):
        with open(manifest_path) as r:
            m = json.load(r)

        self.shards = m["shards"]
        self.cums = m["cumulative_sizes"]
        self.total = m["num_batches"]
        self.fkey = m["feature_key"]
        self.return_meta = return_meta
        self.sticky = sticky_handles
        self._handles: Dict[str, Any] = {}  # path -> safe_open handle (per worker)

    def __len__(self):
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_idx = bisect.bisect_right(self.cums, idx)
        prev = self.cums[shard_idx - 1] if shard_idx > 0 else 0
        return shard_idx, idx - prev

    def _get_handle(self, path: str):
        if not self.sticky:
            return safe_open(path, framework="pt")
        h = self._handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt")
            self._handles[path] = h
        return h

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self.total
        if not (0 <= idx < self.total):
            raise IndexError(idx)

        shard_i, local_i = self._locate(idx)
        info = self.shards[shard_i]
        path = info["path"]

        if self.sticky:
            f = self._get_handle(path)
            x = f.get_tensor(self.fkey)[local_i]
        else:
            with self._get_handle(path) as f:
                x = f.get_tensor(self.fkey)[local_i]

        return x


if __name__ == '__main__':
    ds = SafeTensorShards('data/finewebedu_combined/manifest_layer9.json', return_meta=True)
    for i in range(len(ds)):
        print(i, ds.__getitem__(i).shape)
