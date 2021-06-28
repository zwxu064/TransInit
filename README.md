# Reference

- If you find our work, [TransInit](https://arxiv.org/abs/2010.04363), or the [code](https://github.com/zwxu064/TransInit.git) useful, please cite it below.
  ```
  @article{xu2020transinit,
  title={Refining Semantic Segmentation with Superpixel by Transparent Initialization and Sparse Encoder},
  author={Zhiwei Xu, Thalaiyasingam Ajanthan, and Richard Hartley},
  journal={arXiv:2010.04363},
  year={2020}
  }
  ```
 
 # Environment
 One should have at least 4 GPUs with each 11 GB (batch size is 16 for reproducing the baseline) for jointly fine-tuning
 both semanic segmentation (ReNet152 used in our work) and superpixel FCN.
 For validation, 1 GPU with 11 GB would be sufficient.
 
  ```
  # For the main dependencies
  conda create -n TransInit python=3.7.1
  source activate TransInit
  conda install pytorch=1.2.0 torchvision=0.4.0 cudatoolkit -c pytorch
  
  # For other dependencies, one should install according to the virtual environment.
  # We provide the versions used in ours.
  tqdm=4.35.0
  tensorboardx=1.8
  setuptools=41.4.0
  scipy=1.3.1
  scikit-image=0.16.2
  pillow=6.1.0
  numpy=1.17.1
  matplotlib=3.1.1
  imageio=2.8.0
  
  # Alternative
  bilateralfilter=0.1 (one needs to build this according to the guidance in RLoss for semantic segmentation with ResNet backbone)
  ```
 
 # How to Use
The two core features of this work are **transparant initialization** for an identical mapping and **sparse encoder** for an index transition between superpixels and pixels.

The **transparent initialization** module is in "joint_learning/trans_init.py" with a simple demo in "unittest/test_trans_init.py"
while the **sparse encoder** module applied to superpixel FCN network is distributed in "joint_learning/spnet.py".

- For **dataset**, in this ablation study, we used PASCAL VOC combined with Berkeley benchmark for training and validation.
  Please download Berkeley benchmark and PASCAL VOC 2012 using the scripts from ["./data/"](https://github.com/meng-tang/rloss.git) and put the merged dataset in "./datasets/PASCAL".
  It should contain folders such as "ImageSets", "JPEGImages", "SegmentationClassAug", etc.
  Here, we provide the validation set list for convenience, see "datasets/PASCAL/val.txt".

- For **pretrained models**, we provide the one for PASCAL VOC validation as a **demo**.
One can refer to the usage of transparent initialization and sparse encoder modules in this demo for other tasks.
Please download [it](https://1drv.ms/u/s!AngC1-tRlyPMgk9MTXx8x-8CVLvK?e=mItEAr) to "pretrained" and set the corresponding path in "train.sh" and "evaluation.sh".

- For **training**, run the following script, one will find a bash script in "checkpoints/scripts/pascal"
  ```
  ./train.sh
  ```

- For **validation**, run the following script with a bash script in "checkpoints/scripts/pascal/eval"
  ```
  ./evaluation.sh
  ```

# Notes
In this repository, the main deep learning backbone for **semantic segmentation** is ResNet-serial on PASCAL VOC, as an ablation study in our work, see "third_party/rloss".
One can find its original code from [RLoss](https://github.com/meng-tang/rloss.git).
If you use "third_party/rloss" in your work, please cite [the paper](https://cs.uwaterloo.ca/~m62tang/OnRegularizedLosses_ECCV18.pdf).

For **superpixel network**, we used [superpixel FCN](https://github.com/fuy34/superpixel_fcn.git).
If you use "third_party/sp_fcn" in your work, please cite [the paper](https://arxiv.org/abs/2003.12929).

If you have any questions, please contact zhiwei.xu@anu.edu.au.