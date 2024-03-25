# Single Path Global Modulation (SPGM)

This repository implements SPGM as a standalone model based on our [paper](https://arxiv.org/abs/2309.12608) accepted by ICASSP 2024

## Demo

A demo with instructions on how to run inference on the model is available as a colab notebook [here](https://colab.research.google.com/drive/1zKEaRFNITve7WPsqVNUuaRXiduR7H1Ki?usp=sharing)

Training is handled by speechbrain. This can be done through my fork of the speechbrain repository found [here](https://github.com/Yip-Jia-Qi/speechbrain/tree/add_spgm).

## Results
Here are the SI - SNRi results (in dB) on the test set of WSJ0-2 Mix:

|Model| Data Augmentation | WSJ0-2Mix (SI-SNRi)|
| --- |--- | --- |
|spgm (paper)|SpeedPerturb | 22.1 |
|[spgm-base](https://huggingface.co/yipjiaqi/spgm-base)|DynamicMixing | 22.7 |
|[spgm-opt](https://huggingface.co/yipjiaqi/spgm-opt)|DynamicMixing | 23.0 |

In the original paper accepted to ICASSP, the only data augmentation used was speed perturbation. Subsequently we trained the model using dynamic mixing, which yielded improvements in performance. 

Additionally, after further exploring some hyperparameters, we obtain an optimized version of SPGM, spgm-opt that achieved 23.0dB SI-SDRi

The weights and config of spgm-base and spgm-opt have been uploaded to huggingface and can be accessed using the code in the repo.

## Citation

Please cite our paper if you have found this model useful
```bibtex
@INPROCEEDINGS{yip2023spgm,
  author={Yip, Jia Qi and Zhao, Shengkui and Ma, Yukun and Ni, Chongjia and Zhang, Chong and Wang, Hao and Nguyen, Trung Hieu and Zhou, Kun and Ng, Dianwen and Chng, Eng Siong and others},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={SPGM: Prioritizing Local Features for enhanced speech separation performance},
  year={2024},
}
```
