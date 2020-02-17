# 百度AI安全对抗赛 solution

本项目基于飞桨PaddlePaddle实现。

项目内容为百度AI安全对抗赛代码，基于官方给出的PGD修改，主要内容为L2-PGD+EOT。


## Usage

比赛中使用的攻击思想已放入[attack_demo.ipynb](https://github.com/persistz/baidu_ai_attack_SSC/blob/master/attack_demo.ipynb "attack_demo.ipynb")

如需要复现完整攻击，请下载如下数据集：
[AIStudio链接](https://aistudio.baidu.com/aistudio/datasetdetail/19743)并参考目录src下的文件


## Main Idea

- 使用$L_2$ norm约束对抗样本的扰动
- 合理定义loss函数实现confidence较高的对抗样本

更多细节可参考 write up


## Related Links

比赛链接：[https://aistudio.baidu.com/aistudio/competition/detail/15](https://aistudio.baidu.com/aistudio/competition/detail/15)

PaddlePaddle官方文档：
[https://www.paddlepaddle.org.cn/](https://www.paddlepaddle.org.cn/)



