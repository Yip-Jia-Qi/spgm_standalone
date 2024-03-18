from huggingface_hub import PyTorchModelHubMixin
import torch
from model.SPGM import SPGMWrapper
from model.SPGM_configs import spgm_base

#This code should only need to be run once

# create model
model = SPGMWrapper()
model.loadPretrained()
print("pretrained model loaded")

# save locally
model.save_pretrained("spgm-base", config=spgm_base)

# push to the hub
model.push_to_hub("yipjiaqi/spgm-base", config=spgm_base)

model = SPGMWrapper.from_pretrained("yipjiaqi/spgm-base")

out = model.inference("./test_samples/item0_mix.wav","./test_samples/")
print(out)



