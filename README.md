# CREnhancer_Low-light_Enhancemnt
Code of Perceiving Image via Decomposing: A Component Regularization-based Low-light Image Enhancer

#### Recommended Environment:<br>
 - [ ] python = 2.7
 - [ ] tensorflow-gpu = 1.9.0
 - [ ] numpy = 1.15.4
 - [ ] scipy = 1.2.0
 - [ ] pillow = 5.4.1
 - [ ] scikit-image = 0.13.1
 
 #### Prepare training data :<br>
- [ ] Download the LOL dataset from [here](https://daooshee.github.io/BMVC2018website/) or [here](https://drive.google.com/open?id=1-MaOVG7ylOkmGv1K4HWWcrai01i_FeDK). Put [I<sub>low</sub>,I<sub>high</sub>] in the "./low/..." and "./high/..." for training decomposition network.
- [ ] Put paired or unpaired [I<sub>low</sub>,I<sub>well</sub>] in the "./low/..." and "./well/..." for training Enhancement network.

#### Training :<br>
- [ ] Run "CUDA_VISIBLE_DEVICES=X python DNet_train.py" to train decomposition network.
- [ ] Run "CUDA_VISIBLE_DEVICES=X python EnhanceGAN_train.py" to train enhancement network.

#### Testing :<br>
- [ ] Run "CUDA_VISIBLE_DEVICES=X python evaluate_EnhanceGAN.py" to enhance the provided low-light images.
- [ ] You can also visualize the decomposed components by running "CUDA_VISIBLE_DEVICES=X python evaluate_DNet.py".
