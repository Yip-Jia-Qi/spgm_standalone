# Single Path Global Modulation (SPGM)

(This is the hf_compatible branch which is meant for uploading the model to huggingface)

This repository implements SPGM as a standalone model based on our [paper](https://arxiv.org/abs/2309.12608) accepted by ICASSP 2024

The config is preset and weights are provided based on the model that was trained on WSJ0-2Mix with Dynamic Mixing.

Here are the SI - SNRi results (in dB) on the test set of WSJ0-2 Mix:

| | SPGM, WSJ0-2Mix |
|--- | --- |
|SpeedPerturb | 22.1 |
|DynamicMixing | 22.7 |

A demo with instructions on how to run inference on the model is available as a colab notebook [here](https://colab.research.google.com/drive/1zKEaRFNITve7WPsqVNUuaRXiduR7H1Ki?usp=sharing)

Please cite our paper if you have found this model useful
```bibtex
@INPROCEEDINGS{yip2023spgm,
  author={Yip, Jia Qi and Zhao, Shengkui and Ma, Yukun and Ni, Chongjia and Zhang, Chong and Wang, Hao and Nguyen, Trung Hieu and Zhou, Kun and Ng, Dianwen and Chng, Eng Siong and others},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={SPGM: Prioritizing Local Features for enhanced speech separation performance},
  year={2024},
}
```
