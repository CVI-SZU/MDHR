# Multi-Scale Dynamic and Hierarchical Relationship Modeling for Facial Action Units Recognition


Paper: https://arxiv.org/abs/2404.06443

## Overview:
[overview.pptx](https://github.com/user-attachments/files/15531902/overview.pptx)


## Requirements
- Python3
- PyTorch

## Training
python train.py --backbone resnet --fold 1 --dataset_path /path/to/BP4D_dataset/ 

## Test
python test_BP4D.py --backbone resnet --fold 1 --dataset_path /path/to/BP4D_dataset/ --resume /path/to/best_model_fold1.pth --evaluate

## results

## Citation
if the code or method help you in the research, please cite the following paper:
```bash
@article{wang2024multi,
  title={Multi-scale Dynamic and Hierarchical Relationship Modeling for Facial Action Units Recognition},
  author={Wang, Zihan and Song, Siyang and Luo, Cheng and Deng, Songhe and Xie, Weicheng and Shen, Linlin},
  journal={arXiv preprint arXiv:2404.06443},
  year={2024}
}
```
## Acknowledgements
This repo is built using components from  [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU)

