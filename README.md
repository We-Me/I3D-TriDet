# 挑战性课程——基于Transformer架构的视频片段查找与分类

## 项目简介
本项目旨在实现一个基于Transformer架构的视频片段查找与分类模型，用于对给定视频进行动作片段的定位和分类。该项目结合了 I3D(Inflated 3D ConvNet) 视频特征提取和近期提出的 [TriDet](https://github.com/dingfengshi/TriDet "[CVPR2023] TriDet: Temporal Action Detection with Relative Boundary Modeling") 模型。简单来说，模型会从视频中提取时序特征，再利用Transformer模型对视频中的动作发生段进行定位，并给出动作类别。项目面向课程老师/助教作为课程作业展示，重点体现方法的实现过程和基本功能，无过多市场宣传或面向开发者的细节。

## 环境依赖
- Python 3.8  
- PyTorch 2.2.1（CUDA 11.8）  
- 其他依赖库见 `requirements.txt`

## 安装与运行方法

按照以下步骤配置环境并运行项目代码：

1. 安装PyTorch：请先确保安装了匹配CUDA版本的PyTorch。例如，对于CUDA 11.8，可使用以下命令安装 PyTorch 2.2.1：
  
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. 安装依赖库：在项目根目录下运行命令安装所需的Python依赖：

```shell
pip install -r requirements.txt
```

3. 编译NMS模块：进入NMS代码目录并编译安装1D非极大值抑制模块（用于后处理过滤冗余检测片段）：

```shell
cd ./TriDet/libs/utils
python setup.py install --user
cd ../..
```

4. 准备预训练模型权重：下载或获取I3D模型的预训练权重文件，并放置到i3d_feature_extraction/models/目录下。代码默认使用两个文件：rgb_imagenet.pt（RGB帧I3D模型）和flow_imagenet.pt（光流I3D模型）。这些权重用于提取视频的时序特征。

5. 准备数据集：若需要在特定数据集上训练或评估模型，请按配置要求准备数据。例如，THUMOS14数据集需准备标注文件和预提取的I3D特征文件放入data/thumos/对应目录下。

完成以上环境配置后，可以按照以下方式使用本项目：

- 模型训练：使用提供的训练脚本可在支持的数据集上进行训练。以在THUMOS14数据集上训练为例，确保已准备好特征和标注，然后运行：

```shell
python TriDet/train.py TriDet/configs/thumos_i3d.yaml
```

- 模型推理：本项目提供了脚本用于对新视频进行动作片段检测和可视化。将待测试的MP4格式视频文件放入tmp/video/目录，然后运行：

```shell
python vedio2predict.py
```

## 模型说明

**特征提取**：本项目采用双流I3D模型对视频进行特征提取。I3D(Inflated 3D ConvNet)是一种将2D卷积拓展为3D卷积的深度网络，可用于视频动作识别。我们分别使用RGB图像帧和光流场作为I3D的输入，得到两路时序特征向量。每段视频由I3D提取出固定维度的特征序列，这些特征序列将作为后续Transformer模型的输入。

**动作片段检测**：我们实现并采用了TriDet模型来对提取的特征序列进行动作片段定位与分类。TriDet是一种基于 Transformer 的时序动作检测模型，其特点是在Transformer架构中引入相对边界建模(Relative Boundary Modeling)机制。简单来说，模型以视频特征序列为输入，利用自注意力机制学习时序上下文信息，并通过边界回归分支预测动作段的相对起始和结束边界，从而定位出视频中的候选动作片段。与此同时，模型的分类分支会为每个候选片段预测一个动作类别及其置信度。

**后处理与可视化**：模型输出的候选片段经过NMS(非极大值抑制)处理，以过滤掉重叠冗余的检测结果，只保留高置信度的片段。最终的检测结果包含若干视频时间段以及对应的动作类别标签。项目提供了简单的可视化界面：利用OpenCV将检测到的片段区间和类别名称绘制在原视频上，方便直观地查看模型识别到的动作片段。
