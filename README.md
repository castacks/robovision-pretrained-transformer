# Pre-Trained Vision Transformer for Robot Vision Tasks

>_Research Project at CMU Robotics_

This Pre-Trained Vision Transformer explores how neural networks truly learn information and proposes a novel training method to improve generalization across multiple robot vision tasks — optical flow, disparity, and depth estimation.

## Concept

The project removes the task-specific head from a Vision Transformer (ViT) and introduces a new training algorithm focused on matching the feature maps of the feature-extractor portion of the network.
This forces the ViT to learn robust pixel-matching representations during pre-training instead of relying on task-specific optimizations.

## Methodology
1.	**Model Redesign:** The task-specific head at the end of the ViT was removed.
2.	**Feature Matching Objective:** Training was reformulated around matching internal feature maps between datasets.
3.	**Dataset Conversion:** A custom method was developed to convert optical flow datasets into compatible datasets for this training process.
4.	**Generalization Hypothesis:** By emphasizing feature-level consistency over task-specific loss, the network learns more general and transferable representations.


>_The Unimatch models serves as the base model for this project:_

>@article{xu2023unifying,
>  title={Unifying Flow, Stereo and Depth Estimation},
>  author={Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Yu, Fisher and Tao, Dacheng and Geiger, Andreas},
>  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
>  year={2023}
>}
