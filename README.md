# MDHR
\[CVPR2024\]Code for paper 'Multi-Scale Dynamic and Hierarchical Relationship Modeling for Facial Action Units Recognition'

# Train
python train.py --backbone resnet --fold 1 --dataset_path /path/to/BP4D_dataset/ 

# Test
python test_BP4D.py --backbone resnet --fold 1 --dataset_path /path/to/BP4D_dataset/ --resume /path/to/best_model_fold1.pth --evaluate