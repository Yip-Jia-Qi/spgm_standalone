##This is a sample inference script to demonstrate how to run inference on the model for a single .wav file

from SPGM import SPGMWrapper

model = SPGMWrapper()
model.loadPretrained()
print("pretrained model loaded")

out = model.inference("./test_samples/item0_mix.wav","./test_samples/")
print(out)