# concerto
## 环境：
镜像地址：tqcr/tangxb/concerto:mcore-0.12.2 
算法库：concerto  

## 训练脚本：
两个简单的示例：  
cpt:
```
cd examples/qwen
bash qwen3_cpt.sh
``` 
sft:
```
cd examples/qwen
bash qwen3_sft.sh
```
在示例中给出的是qwen3在32k数据长度下使用TP=4, EP=8训练qwen3-30b-a3b的示例脚本
以下常见的配置可通过用户设置环境变量来配置：
模型权重path：PRETRAIN_CHECKPOINT_PATH
数据path: DATASET_PATH
数据长度： SEQ_LEN
是否采用online_packing: ONLINE_PACKING (true/false)
重计算： AC (full, selective, none)
并行配置：TP, PP, EP, CP, SP


## 权重转换脚本：
qwen2
```
cd toolkits/model_checkpoints_convertor
bash convert_model.sh
```
qwen3
```
cd toolkits/distributed_checkpoints_convertor
bash scripts/qwen3/convert_example.sh
```

## online tokenize+packing配置和逻辑介绍
* 启动脚本增加参数`--online-packing`， 目前在qwen_sft, qwen_cpt里已默认生效  
* 使用该配置后，会重写dataloader逻辑，这部分逻辑封装为build_data_loader函数，可由用户自己定义，新的dataloader在迭代datasets过程时会在后台持续进行tokenize+packing的过程，不会阻塞住正常的训练过程，生成的结果会存入队列里，队列的长度由dataloader的参数prefetch_factor控制  
* 用户可根据自己需求自定义tokenize和packing的具体实现逻辑，通过重写dataloader的`collate_fn`和`concat_fn`实现，目前已经给出了sft和cpt的默认方案
