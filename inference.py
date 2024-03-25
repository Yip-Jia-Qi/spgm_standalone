'''This is a sample inference script to demonstrate how to run inference on the model for a single .wav file

Authors
* Jia Qi Yip 2024
'''
from model.SPGM import SPGMWrapper

model_configs = ["spgm-base", "spgm-opt"]

for mc in model_configs:
    model = SPGMWrapper.from_pretrained(f'yipjiaqi/{mc}')
    print("pretrained model loaded")

    out = model.inference("./test_samples/item0_mix.wav",f'./test_samples/{mc}')
    print(out)