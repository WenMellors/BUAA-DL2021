## 中作业任务说明

中作业将基于课程组开发的交通预测领域开源框架，要求同学们以两人一组（**不能超过2人，不推荐单人组队**）的形式选择一开源模型将其改写为符合框架接口约束的模型。

具体来说，课程组预先选定一批交通开源模型——涉及交通流量/速度/需求量预测、轨迹下一跳预测。并且，课程组前期已经为前述任务准备好了数据集并搭建了数据预处理、评估模块。因此，各小组的主要工作是将模型开源代码改写为符合框架抽象接口约束的模型类，虽然课程组已经实现了大部分通用的数据接口，但是**部分模型可能需要对数据接口进行一定的修改**。

#### 作业分数构成

中作业占总成绩的 10%（10分），由以下四部分组成：

* 复现模型完成度（4分）：要求复现的模型能够在框架中运行。
* 技术报告（4分）：具体要求参见技术报告模板。
* 复现性能（2分）：考量复现模型的性能。
* 难易度加分（附加分）：考虑到部分难度较高的模型，复现难度较高，因此完成中等难度模型复现工作（即能在框架中运行）的小组可获得 1 分额外加分，完成困难模型复现工作的小组获得 2 分额外加分。（注：若加分后总分超过 10 分，则按 10 分计算）。

## 框架介绍

![](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/_images/pipeline.png)

框架以流水线的形式运行，主要分为五个步骤：

1. 初始化流水线配置。（依托 Config 模块）
2. 数据集加载与数据预处理，数据转换并划分训练集、验证集、测试集。（依托 Data 模块）
3. 加载模型。（依托 Model 模块）
4. 训练验证模型，并在测试集上进行测试。（依托 Executor 模块）
5. 评估模型测试输出。（依托 Evaluator 模块）

各组的工作主要涉及 Model 模块，通过使用课程组预先构建的 Data 模块提供的输入数据进行任务预测，并输出符合课程组预先构建 Evaluator 模块的评估输入接口格式。对于 POI 轨迹下一跳预测任务与交通流量\速度预测任务具体接口格式的说明，可参见对应任务的说明 md 文件。

关于框架的具体介绍可以参考[文档](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/index.html)。

## 框架使用

课程组预先构建了两个脚本文件，方便各组测试运行复现后的模型。

#### test_model.py

测试模型是否能跑通框架的脚本文件，各组可以参考该脚本文件在调式模式或命令行中测试运行模型，与通过实操完成对框架的深入理解。

#### run_model.py

训练模型并在测试集上进行预测，最后会将测试结果输出至命令行与 `trafficdl/cache/evaluate_cache/` 文件夹下。

命令行运行示例：

```sh
python run_model.py --task traj_loc_pred --model DeepMove --dataset foursquare_tky
```

这里简单介绍部分常用命令行参数：

* task：所要执行的任务名，默认为`traj_loc_pred`。需要各组修改为自己对应的任务名。
* model：所要运行的模型名，默认为`DeepMove`。需要各组修改为自己对应的模板模型名。
* dataset：所要运行的数据集，默认为 `foursquare_tky`。
* config_file：用户指定 config 文件名，默认为 `None`。
* saved_model：是否保存训练的模型结果，默认为 `True`。
* train：当模型已被训练时是否要重新训练，默认为 `True`。

## 论文列表

#### POI 轨迹下一跳预测（POI 推荐）

|      | 模型名         | 难度 | 论文                                                         | 开源代码                                                     |
| ---- | -------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | STRNN          | 中   | [Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPDFInterstitial/11900/11583) | [pytorch](https://github.com/yongqyu/STRNN)                  |
| 2    | LSTPM          | 中   | [Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/5353) | [pytorch](https://github.com/NLPWM-WHU/LSTPM)                |
| 3    | GeoSAN         | 难   | [Geography-Aware Sequential Location Recommendation](https://dl.acm.org/doi/pdf/10.1145/3394486.3403252) | [pytorch](https://github.com/libertyeagle/GeoSAN)            |
| 4    | Flashback(RNN) | 中   | [Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States](https://www.ijcai.org/Proceedings/2020/0302.pdf) | [pytorch](https://github.com/eXascaleInfolab/Flashback_code) |
| 5    | ATST-LSTM      | 难   | [An Attention-based Spatiotemporal LSTM Network for Next POI Recommendation](https://ieeexplore.ieee.org/abstract/document/8723186) | [tensorflow](https://github.com/drhuangliwei/An-Attention-based-Spatiotemporal-LSTM-Network-for-Next-POI-Recommendation) |
| 6    | STAN           | 中   | [STAN: Spatio-Temporal Attention Network for Next Location Recommendation](https://arxiv.org/pdf/2102.04095v1.pdf) | [pytorch](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation) |
| 7    | STF-RNN        | 易   | [STF-RNN: Space Time Features-based Recurrent Neural Network for Predicting People Next Location](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_SSCI_2016/pdf/SSCI16_paper_377.pdf) | [keras](https://github.com/mhjabreel/STF-RNN)                |
| 8    | CARA           | 难   | [A Contextual Attention Recurrent Architecture for Context-Aware Venue Recommendation](https://dl.acm.org/doi/10.1145/3209978.3210042) | [keras](https://github.com/feay1234/CARA)                    |

#### 交通状态预测（流量、速度、需求量）

流量

| 编号 | 模型名        | 难度 | 论文                                                         | 开源代码                                                    |
| ---- | ------------- | ---- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| 9    | ST-MetaNet    | 难   | [Urban traffic prediction from spatio-temporal data using deep meta learning](https://dl.acm.org/doi/abs/10.1145/3292500.3330884) | [MXNet](https://github.com/panzheyi/ST-MetaNet)             |
| 10   | STSGCN        | 中   | [Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/5438) | [MXNet](https://github.com/Davidham3/STSGCN)                |
| 11   | DSAN          | 中   | [Preserving Dynamic Attention for Long-Term Spatial-Temporal Prediction](https://dl.acm.org/doi/10.1145/3394486.3403046) | [tf2](https://github.com/hxstarklin/DSAN)                   |
| 12   | ST-GDN        | 难   | [Traffic Flow Forecasting with Spatial-Temporal Graph Diffusion Network](https://www.aaai.org/AAAI21Papers/AISI-9334.ZhangX.pdf) | [tf](https://github.com/jillbetty001/ST-GDN)                |
| 13   | STDN          | 中   | [Revisiting spatial-temporal similarity: A deep learning framework for traffic prediction](https://www.aaai.org/ojs/index.php/AAAI/article/view/4511) | [Keras](https://github.com/tangxianfeng/STDN)               |
| 14   | STFGNN        | 中   | [Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting](https://arxiv.org/abs/2012.09641) | [MXNet](https://github.com/MengzhangLI/STFGNN)              |
| 15   | STNN          | 易   | [Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relations Discovery](https://ieeexplore.ieee.org/document/8215543) | [Pytorch](https://github.com/edouardelasalles/stnn)         |
| 16   | STAG-GCN      | 易   | [Spatiotemporal Adaptive Gated Graph Convolution Network for Urban Traffic Flow Forecasting](https://dl.acm.org/doi/abs/10.1145/3340531.3411894) | [Pytorch](https://github.com/RobinLu1209/STAG-GCN)          |
| 17   | ST-CGA        | 中   | [Spatial-Temporal Convolutional Graph Attention Networks for Citywide Traffic Flow Forecasting](https://dl.acm.org/doi/abs/10.1145/3340531.3411941) | [Keras](https://github.com/jillbetty001/ST-CGA)             |
| 18   | ResLSTM       | 中   | [Deep Learning Architecture for Short-Term Passenger Flow Forecasting in Urban Rail Transit](https://ieeexplore.ieee.org/abstract/document/9136910/) | [Keras](https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN) |
| 19   | DGCN          | 易   | [Dynamic Graph Convolution Network for Traffic Forecasting Based on Latent Network of Laplace Matrix Estimation](https://ieeexplore.ieee.org/abstract/document/9190068/) | [Pytorch](https://github.com/guokan987/DGCN)                |
| 20   | Multi-STGCnet | 中   | [Multi-STGCnet: A Graph Convolution Based Spatial-Temporal Framework for Subway Passenger Flow Forecasting](https://ieeexplore.ieee.org/abstract/document/9207049/) | [Keras](https://github.com/start2020/Multi-STGCnet)         |
| 21   | Conv-GCN      | 中   | [Multi-Graph Convolutional Network for Short-Term Passenger Flow Forecasting in Urban Rail Transit](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/iet-its.2019.0873) | [Keras](https://github.com/JinleiZhangBJTU/Conv-GCN)        |
| 22   | TCC-LSTM-LSM  | 中   | [A temporal-aware LSTM enhanced by loss-switch mechanism for traffic flow forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0925231220318130) | [Keras](https://github.com/illumina7e/TCC-LSTM-LSM)         |
| 23   | CRANN         | 易   | [A Spatio-Temporal Spot-Forecasting Framework forUrban Traffic Prediction](https://arxiv.org/abs/2003.13977) | [Pytorch](https://github.com/rdemedrano/crann_traffic)      |

速度

| 编号 | 模型名       | 难度 | 论文                                                         | 开源代码                                                     |
| ---- | ------------ | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 24   | BaiduTraffic | 难   | [Deep sequence learning with auxiliary information for traffic prediction](https://dl.acm.org/doi/abs/10.1145/3219819.3219895) | [tf](https://github.com/JingqingZ/BaiduTraffic)              |
| 25   | GMAN         | 难   | [Gman: A graph multi-attention network for traffic prediction](https://www.aaai.org/ojs/index.php/AAAI/article/view/5477) | [tf](https://github.com/zhengchuanpan/GMAN)                  |
| 26   | MRA-BGCN     | 易   | [Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting](https://arxiv.org/ftp/arxiv/papers/1911/1911.12093.pdf) | [Pytorch](https://github.com/wumingyao/MAR-BGCN_GPU_pytorch) |
| 27   | FC-GAGA      | 难   | [FC-GAGA: Fully Connected Gated Graph Architecture for Spatio-Temporal Traffic Forecasting](https://arxiv.org/abs/2007.15531) | [tf](https://github.com/boreshkinai/fc-gaga)                 |
| 28   | HGCN         | 易   | [Hierarchical Graph Convolution Networks for Traffic Forecasting](https://github.com/guokan987/HGCN/blob/main/paper/3399.GuoK.pdf) | [Pytorch](https://github.com/guokan987/HGCN)                 |
| 29   | GTS          | 易   | [Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://arxiv.org/pdf/2101.06861) | [Pytorch](https://github.com/chaoshangcs/GTS)                |
| 30   | DKFN         | 易   | [Graph Convolutional Networks with Kalman Filtering for Traffic Prediction](https://dl.acm.org/doi/abs/10.1145/3397536.3422257) | [Pytorch](https://github.com/Fanglanc/DKFN)                  |
| 31   | GaAN         | 难   | [GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal Graphs](https://arxiv.org/abs/1803.07294) | [MXNet](https://github.com/jennyzhang0215/GaAN)              |
| 32   | ST-MGAT      | 易   | [ST-MGAT: Spatial-Temporal Multi-Head Graph Attention Networks for Traffic Forecasting](https://ieeexplore.ieee.org/abstract/document/9288309) | [Pytorch](https://github.com/Kelang-Tian/ST-MGAT)            |
| 33   | DGFN         | 中   | [Dynamic Graph Filters Networks: A Gray-box Model for Multistep Traffic Forecasting](https://ieeexplore.ieee.org/abstract/document/9294627/) | [tf2](https://github.com/RomainLITUD/DGCN_traffic_forecasting) |
| 34   | ATDM         | 易   | [On the Inclusion of Spatial Information for Spatio-Temporal Neural Networks](https://arxiv.org/abs/2007.07559) | [Pytorch](https://github.com/rdemedrano/SANN)                |

需求

| 编号 | 模型名    | 难度 | 论文                                                         | 开源代码                                                     |
| ---- | --------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 35   | DMVST-Net | 中   | [Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction](https://arxiv.org/abs/1802.08714) | [Keras](https://github.com/huaxiuyao/DMVST-Net)              |
| 36   | STG2Seq   | 难   | [Stg2seq: Spatial-temporal graph to sequence model for multi-step passenger demand forecasting](https://arxiv.org/abs/1905.10069) | [tf](https://github.com/LeiBAI/STG2Seq)                      |
| 37   | CCRNN     | 易   | [Coupled Layer-wise Graph Convolution for Transportation Demand Prediction](https://arxiv.org/abs/2012.08080) | [Pytorch](https://github.com/Essaim/CGCDemandPrediction)     |
| 38   | SHARE     | 易   | [Semi-Supervised Hierarchical Recurrent Graph Neural Network for City-Wide Parking Availability Prediction](https://ojs.aaai.org/index.php/AAAI/article/download/5471/5327) | [Pytorch](https://github.com/Vvrep/SHARE-parking_availability_prediction-Pytorch) |
| 39   | PVCGN     | 易   | [Physical-Virtual Collaboration Modeling for Intra-and Inter-Station Metro Ridership Prediction](https://arxiv.org/abs/2001.04889) | [Pytorch](https://github.com/ivechan/PVCGN)                  |

## 数据集

1. POI 轨迹下一跳预测使用 foursqaure-tky 数据集[下载链接](https://bhpan.buaa.edu.cn:443/link/55CCED27725C7FDD6EEEE8BCEEDCF63F)。

请将下载好的数据集存放于 `code/raw_data` 文件夹下。

2. 交通状态预测数据集[下载链接](https://bhpan.buaa.edu.cn:443/link/E3DB96256D8E99FB2B29B864E92F123A)

**数据集和模型的对应关系查看文件`交通状态数据集和模型对应关系.xlsx`**

请将下载好的数据集存放于 `code/raw_data/数据集名/数据集具体的文件` 文件夹下。（直接下载数据集对应的文件夹，解压到`code/raw_data/`下即可。）

