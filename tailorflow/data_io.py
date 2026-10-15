import os
import numpy as np
import torch
from tailorflow.preprocessing.smpl import DATASET

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
C3D_ROOT = os.environ.get('C3D_ROOT', os.path.join(DATA, DATASET))
RENDER_PICKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render_picks.json')


class MeshClothDataset(torch.utils.data.Dataset):
    """C3D items {root}/{split}/{category}/<id>.npz written by preprocessing/build.py."""

    def __init__(self, categories, split, limit=0, root=C3D_ROOT):
        self.items = []
        self.cate_to_idx = {c: i for i, c in enumerate(categories)}
        for c in categories:
            cdir = os.path.join(root, split, c)
            stems = sorted(f[:-4] for f in os.listdir(cdir) if f.endswith('.npz') and '.' not in f[:-4])
            if limit > 0:
                stems = stems[:limit]
            self.items += [(c, cdir, stem) for stem in stems]
        self.cache = [None] * len(self.items)

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        cate, cdir, stem = self.items[i]
        z = np.load(os.path.join(cdir, f'{stem}.npz'))
        item = {
            'tr_points': torch.from_numpy(z['points']), 'body_verts': torch.from_numpy(z['body']),
            'body_normals': torch.from_numpy(z['body_normals']),
            'body_occupancy': torch.from_numpy(z['occ'].astype(np.float32)).unsqueeze(-1),
            'cate_idx': torch.tensor(self.cate_to_idx[cate]), 'idx': torch.tensor(i), 'beta': torch.from_numpy(z['betas']),
            'sid': cate, 'mid': stem,
        }
        self.cache[i] = item
        return item

    def __getitem__(self, i):
        return self.cache[i] if self.cache[i] is not None else self._load(i)


def build_mesh_loader(categories, split, batch_size, limit=0, num_workers=0, distributed=False, world_size=1, rank=0, wrap=None):
    ds = MeshClothDataset(categories, split, limit=limit)
    if wrap is not None:
        ds = wrap(ds)
    train = split == 'train'
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=train) if distributed else None
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler, drop_last=train,
                                         num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True)
    return loader, sampler
