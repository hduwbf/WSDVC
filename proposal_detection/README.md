## Proposal Generation via Distillation Learning
代码与数据在du@yq01-rp-nlp-rd2.yq01.baidu.com机器上

1 数据

在/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-proposal-part/wsdvc_proposal_part/data文件夹中的action、highlight、activity和event文件夹中保存了THUMOS14、Baidu-Highlight和ActivityNet的C3D特征和proposal标注。

2 教师网络

2.1 三个教师网络的训练代码分别在/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-proposal-part/wsdvc_proposal_part/action_teacher、/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-proposal-part/wsdvc_proposal_part/activity_teacher和/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-proposal-part/wsdvc_proposal_part/highlight_teacher中，训练和测试的方法在每个子文件夹中的README.md中

2.2 三个学生网络训练结束之后将每个教师网络的checkpoint文件夹中的BMN_教师名字.pth.tar文件复制到/home/disk1/du/wubofeng/baidu/personal-code/wsdvc-proposal-part/wsdvc_proposal_part/event_student文件夹的checkpoint中

3 学生网络

学生网络的训练和测试方法在event_student文件夹下的README.md中