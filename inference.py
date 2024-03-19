'''This is a sample inference script to demonstrate how to run inference on the model for a single .wav file

Authors
* Jia Qi Yip 2024
'''
import torch
from model.SPGM import SPGMWrapper

model = SPGMWrapper.from_pretrained("yipjiaqi/spgm-base")
print("pretrained model loaded")

print(model.device)
out = model.inference("./test_samples/item0_mix.wav","./test_samples/")
print(out)