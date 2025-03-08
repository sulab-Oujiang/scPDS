# scPDS
A deep neural network (DNN) framework integrated with transformers to model pathway-based expression profiles, enabling the prediction of drug responses using both bulk and scRNA-seq data. 
![描述](./Workflow.png)
## Hardware
（1）Windows
* 支持的版本：Windows 10 及更高版本，推荐操作系统版本：22000.2538，Windows 11 
* 推荐配置： 
  * CPU：支持 AVX2 指令集的多核处理器 
  * 内存：推荐 16GB RAM 
  * 显卡：支持 CUDA 的 NVIDIA 显卡（至少 4GB VRAM） 
  * 硬盘：至少 500GB SSD（推荐1TB SSD）
  
（2）Linux（推荐操作系统版本：Ubuntu 20.04 或 CentOS 7）

（3）GPU环境
* 推荐使用 NVIDIA Tesla V100、A100 或 RTX 30 系列显卡。
* 安装 CUDA、cuDNN 以及 PyTorch 深度学习框架的 GPU 加速支持。
## Software
* Python 3.8.13
* torch 1.12.1
* argparse 1.1
* numpy 1.24.3
* pandas 1.5.3
* sklearn 1.0.rc2
* imblearn 0.10.1
* matplotlib 3.7.1
## Data Preparation
Data download : https://drive.google.com/file/d/1jec7E9exzZIZqJvddlMIGZkwpDLyfQk4/view?usp=drive_link

文件 scPDS.zip 包含所有的测试数据集。请解压该文件，并将主目录设置为 XX/scPDS/。
## How to cite scPDS
Please cite the following manuscript:

Yao Y, Xu Y, Zhang Y, et al. Single Cell Inference of Cancer Drug Response Using Pathway-Based Transformer Network. Small Methods(2025).
## Usage
* Usage：python scPDS.py
```
optional arguments:
  -d Drug                      Name of the drug, the drug names in the file of --Bulk_label_path(-y)
  -y Bulk_label_path           Path to the bulk RNA-Seq label file
  -x Bulk_data_path            Path to the bulk RNA-Seq data file
  -sc Sc_data_path             Path to the single-cell RNA-Seq data file
  -s sample                    Sampling Strategy: NOsampling, DOWNsampling, UPsampling, SMOTEsampling
  -b batch_size                the number of data samples included in each batch
  -p patience                  the period of early stopping
  -lrp lr_patience             the period of learning rate adjustment
  -n num_epochs                the number of iterations
  -tt train_test_size          Proportion of the dataset reserved for testing
  -tv train_valid_size         Proportion of the training set reserved for validation
  -Tlr T_lr                    Learning rate for the Transformer Model
  -Td T_dropout                Dropout rate for the Transformer Model
  -Tw T_weight_decay           Weight decay (L2 regularization) for the Transformer Model
  -Te T_embedding_dim          Dimension of the embedding layer for the Transformer Model
  -Tb T_bottleneck_dim         Dimension of the bottleneck layer for the Transformer Model
  -Th T_heads                  Number of attention heads in the Transformer Model
  -Tl T_layers                 Number of layers in the Transformer Model
  -plr P_lr                    Learning rate for the prediction model
  -pd P_dropout                Dropout rate for the prediction model
  -pw P_weight_decay           Weight decay (L2 regularization) for the prediction model
  -pi P_input_dim              Input dimension for the prediction model
  -ph P_hidden_dim             Dimension of the hidden layers in the prediction model
  -po P_output_dim             Output dimension for the prediction model
  -seed seed                   Random seed for initializing the model and ensuring reproducibility
```
  
  
  

  

























