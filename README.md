# Elastic Weight Consolidation Done Right for Continual Learning (EWC-DR)

<div style='display:flex; gap: 0.25rem; '>
<a href='LICENSE.txt'><img src='https://img.shields.io/badge/License-Apache 2.0-g.svg'></a>
<a href='https://arxiv.org/abs/2603.18596'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

This repository contains the official implementation of our CVPR 2026 paper, "Elastic Weight Consolidation Done Right for Continual Learning."

In this paper:
- We present a gradient‑based analysis of EWC and its variants, offering new insights for building more reliable and effective regularization‑based continual learning algorithms.
- Our analysis reveals a fundamental **weight importance misalignment**: EWC suffers from gradient vanishing, while MAS experiences redundant protection, leading to inferior continual learning performance.
- We propose **EWC‑DR (EWC Done Right)** , an enhancement to vanilla EWC that introduces a simple **Logits Reversal (LR)** operation during importance estimation. EWC‑DR corrects the misalignment and substantially improves performance across continual learning tasks.

## 1.Requisite

This code is implemented in PyTorch, and we perform the experiments under the following environment settings:

- python = 3.11.4
- torch = 2.0.1
- torchvision = 0.15.2
- timm = 0.6.7

The code has been tested on Linux Platform with a GPU (RTX3080 Ti).

If you see the following error, you may need to install a PyTorch package compatible with your infrastructure.

```
RuntimeError: No HIP GPUs are available or ImportError: libtinfo.so.5: cannot open shared object file: No such file or directory
```

For example if your infrastructure only supports CUDA == 11.1, you may need to install the PyTorch package using CUDA11.1.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

If you see the following error, you can resolve it by installing a lower version of timm, such as pip install timm==0.6.7.

```
TypeError: 'PretrainedCfg' object is not subscriptable
```


## 2.Dataset
 * Create a folder `data/`.
 * **CIFAR 100**: will be automatically downloaded to `data/`.
 * **ImageNet-Subset (ImageNet-100)**: retrieve from [link](https://drive.google.com/drive/folders/1RBrPGrZzd1bHU5YG8PjdfwpHANZR_lhJ?usp=sharing).
 * **Tiny-ImageNet**: retrieve from [link](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and format it into PyTorch's `ImageFolder` structure using [this script](https://github.com/youmingzhao91/Tiny-ImageNet-Pytorch/blob/main/formatting_TinyImageNet.py).

After unzipping ImageNet-Subset and Tiny-ImageNet, place them in the `data/` folder and organize the data as follows:
```
├── ImageNet-100
│   ├── imagenet-100
│   │   ├── train
│   │   └── val
│   ├── eval.txt
│   └── train.txt
└── tiny-imagenet-200
    ├── train
    ├── val
    ├── test
    ├── wnids.txt
    └── words.txt
```

## 3.Training
The JSON configuration files in `exps/` are preconfigured for **10-task incremental learning scenarios**. 
You can customize the continual learning settings by modifying `init_cls` and `increment` parameters in these files.

### CIFAR-100

- Big‑start Incremental Setting:
    ```
    python main.py --config=./exps/ewcdr_cifar_bigstart.json
    ```

- Equally-split Incremental Setting:
    ```
    python main.py --config=./exps/ewcdr_cifar_equalsplit.json
    ```

### ImageNet-Subset

- Big-start Incremental Setting: 
    ```
    python main.py --config=./exps/ewcdr_imagesub_bigstart.json
    ```

- Equally-split Incremental Setting:
    ```
    python main.py --config=./exps/ewcdr_imagesub_equalsplit.json
    ```
  
### Tiny-ImageNet

- Big-start Incremental Setting: 
    ```
    python main.py --config=./exps/ewcdr_tinyimg_bigstart.json
    ```

- Equally-split Incremental Setting:
    ```
    python main.py --config=./exps/ewcdr_tinyimg_equalsplit.json
    ```



## 4.Citation

```bibtex
@inproceedings{liu2026elastic,
  title={Elastic Weight Consolidation Done Right for Continual Learning},
  author={Liu, Xuan and Chang, Xiaobin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```


## 5.Reference
We appreciate the following repositories for their contributions of useful components and functions to our work.

- [PyCIL](https://github.com/LAMDA-CL/PyCIL)
- [TSCIL](https://github.com/zqiao11/TSCIL)



