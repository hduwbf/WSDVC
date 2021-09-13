## Cross-modal Matching
数据与代码均在du@yq01-rp-nlp-rd2.yq01.baidu.com机器中

1 数据

数据均在/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-match-part/data文件夹中

2 训练、测试与生成captionning部分需要的数据

代码一共分为两个版本，分别为/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-match-part/match_part_c3d和/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-match-part/match_part_res（论文中为res部分训练得到的结果），由于resnet特征过大，我在这部分实验后就删除了，下载地址为：https://drive.google.com/file/d/1qzMC0XXkK3-lUUfutFsqY9vmRLTx_Vpw/view 。接下来的部分都用C3D特征训练的代码演示：

训练:
```
/home/disk1/du/wubofeng/envs/env/bin/python train.py
```
训练得到的模型保存在/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-match-part/match_part_c3d/provided_models中

测试:
```
/home/disk1/du/wubofeng/envs/env/bin/python eval.py config/anet_coot.yaml provided_models/anet_coot_AB.pth
```
制作数据集:
```
/home/disk1/du/wubofeng/envs/env/bin/python make_ws_data.py config/anet_coot.yaml provided_models/anet_coot_AB.pth
```
## Caption Generation
此部分的代码使用放在了/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-match-part/captioning_part文件夹中了
