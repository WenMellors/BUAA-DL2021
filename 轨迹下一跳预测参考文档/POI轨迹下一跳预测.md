#### POI 模型约束介绍

课程组提供 POI 轨迹下一跳预测模型模板文件，并已将其添加进赛道中。各组需参考开源模型实现，将其改写为基于 Pytorch 的实现，并完成模板文件中的 `__init__, predict, calculate` 三个方法。

```python
class TemplateTLP(AbstractModel):
    '''
    请参考开源模型代码，完成本文件的编写。请务必补写 __init__, predict, calculate_loss 三个方法。
    '''
    def __init__(self, config, data_feature):
        '''
        参数说明：
            config (dict): 配置模块根据模型对应的 config.json 文件与命令行传递的参数，根据 config 初始化模型参数。
            data_feature (dict): 在数据预处理步骤提取到的数据集所属的特征参数，如 loc_size，uid_size 等。详情其中见下文。
        '''

    def predict(self, batch):
        '''
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值见下文。
        返回值:
            score (pytorch.tensor): 对应张量 shape 应为 batch_size * loc_size。这里返回的是模型对于输入当前轨迹的下一跳位置的预测值。
        '''

    def calculate_loss(self, batch):
        '''
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值见下文。
        返回值:
            loss (pytorch.tensor): 可以调用 pytorch 实现的 loss 函数与 batch['target'] 目标值进行 loss 计算，并将计算结果返回。如模型有自己独特的 loss 计算方式则自行参考实现。
        '''

```

补充说明：

* 设现有用户 $u$ 的一条轨迹 $r_1 \rightarrow r_2 \rightarrow r_3 \rightarrow ...\rightarrow r_{n-1} \rightarrow r_n$，我们会将前 $n-1$ 个点作为当前轨迹输入模型，对 $r_n$ 进行预测。而对于 predict 方法的返回值，可以理解为模型对所有候选的 POI 位置进行置信度打分。

#### 模型输入说明

* config: 模型参数的预设值存放在 `/trafficdl/config/model/TemplateTLP.json` 文件中，请各组参考开源代码的参数设置进行补全。

* data_feature 含有以下键：

  * loc_size：数据集中 POI 点的数目。
  * tim_size：时间窗口的大小，单位是小时。
  * uid_size：数据集中用户的数目。
  * loc_pad：补全轨迹所用的 POI 填充值。
  * tim_pad：补全轨迹所用的时间编码。
  * poi_profile：POI 的 Profile 信息。类型为 pandas.Dataframe，包含 POI 对应的经纬度信息，是直接将原始数据中保存 POI 信息的文件读出的结果，示例如下：

  ```
  geo_id,type,coordinates,venue_category_name
  0,Point,"[139.73674,35.658338]",Neighborhood
  1,Point,"[139.70713999999998,35.697790999999995]",Coffee Shop
  2,Point,"[139.691119,35.690712]",Hotel
  3,Point,"[139.726074,35.711962]",Hotel
  4,Point,"[139.758711,35.672271]",Hotel
  ```
  
* batch 含有以下键：
  
  * history_loc：历史轨迹位置信息
  * history_tim：历史轨迹时间信息
  * current_loc：当前轨迹位置信息
  * current_tim：当前轨迹时间信息
  * uid：用户 id
  * target_loc：要预测的下一跳位置信息
  * target_tim：要预测的下一跳时间信息
  

具体各值对应的数据格式，各组可以参见 test_model.py 脚本，实际在命令行运行进行观察验证。此外对于数据处理部分有想要详细了解的组，可以查看 `/trafficdl/data/dataset/trajectory_dataset.py` 和框架[文档](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/index.html)，以及咨询助教。对于部分困难模型，可能需要组自行参考开源代码的数据处理方式，实现一个 dataset 类。

关于补全，是出于提升模型训练效率、实现批量数据投入（即 batch 输入），才需要将不定长的轨迹数据补齐到一样的长度，做成 tensor 投入模型中。补齐操作是在 `trafficdl.data.batch` 类的 `padding` 方法实现的，此外还提供 `get_origin_len` 方法能够获取到补全之前各轨迹的长度。 

#### 任务可能用到的参数

数据处理部分（配置文件存放于 `/trafficdl/config/data/TrajectoryDataset.json`）：

* min_session_len: 切割后的轨迹的最小长度，小于长度的轨迹会被过滤筛去。
* min_session: 用户最少拥有的轨迹数，小于该值的用户会被过滤筛去。
* time_window_size: 切割轨迹所使用的时间窗格大小。
* history_len: 历史轨迹的最大长度。
* batch_size: 一 batch 数据的长度。
* train_rate: 划分训练集的比例。
* eval_rate: 划分验证集的比例。
* history_type: 取值为 splice 或 cut_off。对于 splice，各段历史轨迹会被拼接成一条长的历史轨迹；对于 cut_off  则不会进行拼接，而由于不拼接的情况下，我们无法保证不同数据点之间的历史轨迹数一样，因此不会对历史轨迹进行补全。

模型执行部分（配置文件存放于 `/trafficdl/config/executor/TrajLocPredExecutor.json`）：

* gpu: 是否使用 gpu 进行训练。
* gpu_id: 使用 gpu 的 id。
* learning_rate: 学习率。

评估部分（配置文件存放于`/trafficdl/config/evaluator/TrajLocPredEvaluator.json`）：

* metrics: 评估使用的评估指标，值为一个数组。具体支持的指标参见[文档](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/user_guide/evaluator/traj_loc_pred.html)
* topk: 评估时具体时 top 几。

