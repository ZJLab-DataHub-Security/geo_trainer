# geo_trainer
## 环境：
镜像地址：10.200.99.202:15080/002295/pytorch:24.07-py3  
算法库：geo_trainer  

## 训练脚本：
```
cd examples/qwen
```
两个简单的示例：  
cpt:
```
bash qwen25_7b_4k_cpt_example.sh
``` 
sft:
```
bash qwen25_7b_4k_sft_example.sh
```
需要修改更详细的配置参数，见：qwen_cpt.sh, qwen_sft.sh

## 权重转换脚本：
```
cd toolkits/model_checkpoints_convertor
bash convert_model.sh
```

## online tokenize+packing配置和逻辑介绍
* 启动脚本增加参数`--online-packing`， 目前在qwen_sft, qwen_cpt里已默认生效  
* 使用该配置后，会重写dataloader逻辑，这部分逻辑封装为build_data_loader函数，可由用户自己定义，新的dataloader在迭代datasets过程时会在后台持续进行tokenize+packing的过程，不会阻塞住正常的训练过程，生成的结果会存入队列里，队列的长度由dataloader的参数prefetch_factor控制  
* 用户可根据自己需求自定义tokenize和packing的具体实现逻辑，通过重写dataloader的`collate_fn`和`concat_fn`实现，目前已经给出了sft和cpt的默认方案

## 备注
由于tqlm版本和base版本具体实现细节存在差别，具体包含以下两点：
1. tqlm版本的mcore版本为0.11.0, base版本为0.10.0 (两个megatron版本都内置在了镜像里，根据需要使用不同的megatron)
2. layer_specs上tqlm版本做了input_layernorm和linear_qkv的算子融合，base版没有做

   
* 训练脚本上通过环境变量`USE_TQLM`控制采用tqlm版本或base版本  
* 在权重转换脚本里通过环境变量`USE_TQLM_SPECS`控制采用tqlm版本或base版本

