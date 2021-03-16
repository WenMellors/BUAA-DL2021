## 中作业任务说明

课程组为大家准备了交通预测领域中一些开源模型，每个小组从下文论文列表中选择一个开源模型进行复现。为了简化复现工作量，课程组已预先搭建好各个任务的框架，准备好数据集、数据预处理脚本、评估脚本。各小组的主要工作是将模型开源代码改写为符合框架抽象接口约束的模型类。

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

| 模型名         | 难度 | 论文                                                         | 开源代码                                                     |
| -------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| STRNN          | 中   | [Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPDFInterstitial/11900/11583) | [pytorch](https://github.com/yongqyu/STRNN)                  |
| LSTPM          | 难   | [Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/5353) | [pytorch](https://github.com/NLPWM-WHU/LSTPM)                |
| GeoSAN         | 难   | [Geography-Aware Sequential Location Recommendation](https://dl.acm.org/doi/pdf/10.1145/3394486.3403252) | [pytorch](https://github.com/libertyeagle/GeoSAN)            |
| Flashback(RNN) | 中   | [Location Prediction over Sparse User Mobility Traces Using RNNs: Flashback in Hidden States](https://www.ijcai.org/Proceedings/2020/0302.pdf) | [pytorch](https://github.com/eXascaleInfolab/Flashback_code) |
| ATST-LSTM      | 难   | [An Attention-based Spatiotemporal LSTM Network for Next POI Recommendation](https://ieeexplore.ieee.org/abstract/document/8723186) | [tensorflow](https://github.com/drhuangliwei/An-Attention-based-Spatiotemporal-LSTM-Network-for-Next-POI-Recommendation) |
| STAN           | 中   | [STAN: Spatio-Temporal Attention Network for Next Location Recommendation](https://arxiv.org/pdf/2102.04095v1.pdf) | [pytorch](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation) |
| STF-RNN        | 易   | [STF-RNN: Space Time Features-based Recurrent Neural Network for Predicting People Next Location](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_SSCI_2016/pdf/SSCI16_paper_377.pdf) | [keras](https://github.com/mhjabreel/STF-RNN)                |
| CARA           | 难   | [A Contextual Attention Recurrent Architecture for Context-Aware Venue Recommendation](https://dl.acm.org/doi/10.1145/3209978.3210042) | [keras](https://github.com/feay1234/CARA)                    |

补充说明：

* STRNN 与 LSTPM 两个模型已复现在框架中但复现性能极差，选择这两个模型的小组的主要工作任务将是优化模型的性能。

## 数据集

POI 轨迹下一跳预测使用 foursqaure-tky 数据集[下载链接](https://bhpan.buaa.edu.cn:443/link/55CCED27725C7FDD6EEEE8BCEEDCF63F)。

请将下载好的数据集存放于 `code/raw_data` 文件夹下。