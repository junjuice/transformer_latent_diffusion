import json

import lightning as pl
from lightning.pytorch.loggers import WandbLogger

from tld.lightnings import DenoiserPL

class Config:
    def __init__(self, data: dict):
        for key in data.keys():
            setattr(self, key, data[key])

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    precision="bf16-mixed",
    log_every_n_steps=1000,
    detect_anomaly=True,
    logger=[WandbLogger(project="ntt_diffusion")]
)

with open("config.json") as f:
    config = json.load(f)
config = Config(config)
model = DenoiserPL(config)

trainer.fit(
    model
)