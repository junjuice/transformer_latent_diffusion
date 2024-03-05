from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from tld.bucketeer import Bucketeer
import webdataset as wds
from webdataset.handlers import warn_and_continue as handler
import json
import tld.danbooru as db 
from fractions import Fraction
import os


def identity(x):
    return x

def true(x):
    return True

class MapFn:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, x):
        return {p[2]: x[i] for i, p in enumerate(self.preprocessors)}
    
class MultiFilter():
    def __init__(self, rules, default=False):
        self.rules = rules
        self.default = default

    def __call__(self, x):
        try:
            x_json = x['json']
            if isinstance(x_json, bytes):
                x_json = json.loads(x_json) 
            validations = []
            for k, r in self.rules.items():
                if isinstance(k, tuple):
                    v = r(*[x_json[kv] for kv in k])
                else:
                    v = r(x_json[k])
                validations.append(v)
            return all(validations)
        except Exception:
            return False
        
class SquarePad(torchvision.transforms.Pad):
    def forward(self, x):
        longer = max(x.shape[-2], x.shape[-1])
        if longer > self.padding:
            x = torchvision.transforms.Resize((int(x.shape[-2]*self.padding/longer), int(x.shape[-1]*self.padding/longer)))(x)
        pad = (0, self.padding - x.shape[-1], 0, self.padding - x.shape[-2])
        return F.pad(x, pad, self.padding_mode, self.fill)

    
def setup_data(bsz, img_size, dataset_path, worker_limit, length=6_500_000):
    # SETUP DATASET
    dataset_path = dataset_path
    db.setup()
    preprocessors = [
            ('jpg;png;webp', torchvision.transforms.ToTensor(), 'images'),
            ("__key__", db.get_tags, "captions"),
            ("__key__", db.get_embeddings, "embeddings")
        ]

    map_fn = MapFn(preprocessors)
    dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=handler
    ).shuffle(690, handler=handler).decode(
        "pilrgb", handler=handler
    ).to_tuple(
        *[p[0] for p in preprocessors], handler=handler
    ).map_tuple(
        *[p[1] for p in preprocessors], handler=handler
    ).map(map_fn)
    # SETUP DATALOADER
    cpus = os.cpu_count()
    dataloader = DataLoader(
        dataset, batch_size=bsz, num_workers=min(cpus, worker_limit), pin_memory=True,
        collate_fn=identity
    )

    dataloader_iterator = Bucketeer(dataloader, density=img_size ** 2, factor=32, interpolate_nearest=False, length=length)

    return dataloader, dataloader_iterator

def setup_data_2(bsz, img_size, dataset_path, worker_limit, length=6_500_000):
    # SETUP DATASET
    dataset_path = dataset_path
    db.setup()
    preprocessors = [
            ('jpg;png;webp', torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                SquarePad(img_size)
                ]), 'images'),
            ("__key__", db.get_tags, "captions"),
            ("__key__", db.get_embeddings, "embeddings")
        ]

    map_fn = MapFn(preprocessors)
    dataset = wds.WebDataset(
        dataset_path, resampled=True, handler=handler
    ).shuffle(690, handler=handler).decode(
        "pilrgb", handler=handler
    ).to_tuple(
        *[p[0] for p in preprocessors], handler=handler
    ).map_tuple(
        *[p[1] for p in preprocessors], handler=handler
    ).map(map_fn)
    # SETUP DATALOADER
    cpus = os.cpu_count()
    dataloader = DataLoader(
        dataset, batch_size=bsz, num_workers=min(cpus, worker_limit), pin_memory=True,
        collate_fn=identity
    )

    return dataloader