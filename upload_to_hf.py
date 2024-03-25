from huggingface_hub import PyTorchModelHubMixin
import torch
from model.SPGM import SPGMWrapper
from model.SPGM_configs import spgm_base, spgm_opt

#This code should only need to be run once

# model_configs = [spgm_base, spgm_opt]
model_configs = [spgm_opt]

# create model
for mc in model_configs:
    model = SPGMWrapper(mc)
    model.loadPretrained()
    print("pretrained model loaded")
    inp = torch.rand(1, 160).to(model.device)
    result = model(inp)
    print(result.shape)
    print(f'model okay if torch.Size([1, 160, {mc["masknet_numspks"]}])')


    # save locally
    model.save_pretrained(mc["config_name"], config=mc)

    # push to the hub
    model.push_to_hub(f'yipjiaqi/{mc["config_name"]}', config=mc)

    model = SPGMWrapper.from_pretrained(f'yipjiaqi/{mc["config_name"]}')

    out = model.inference("./test_samples/item0_mix.wav",f'./test_samples/{mc["config_name"]}')
    print(out)



