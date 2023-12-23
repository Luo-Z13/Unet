# Unet-Based WHU-Photogrammetric-AIwork


使用航空影像建筑物数据集(WHU-Aerial-imagery-dataset)(dataset download:http://gpcv.whu.edu.cn/data/building_dataset.html)， 训练一个稳健的Unet 模型，以识别影像中的房屋建筑物。统计精度指标结果如 IoU，dice loss，召回率等。  
1. 了解深度学习 train,test,validation 数据的概念。  
2. 了解语义分割的概念，语义分割与目标检测的区别。  
3. 了解多种精度指标。 
4. 体会调参过程，探究参数设置和训练周期对训练精度的影响。
5. 尝试使用 tensorboard --logdir=runs 观察训练效果（loss 变化等指标）

可视化结果：


![whubd测试结果1](https://user-images.githubusercontent.com/72430633/196582661-c9163076-7f9a-4259-ba0d-31e0e985cc68.png)
![whubd测试结果2](https://user-images.githubusercontent.com/72430633/196582676-804b7557-e707-4360-b38e-e1d25e59dca5.png)
![loss可视化](https://user-images.githubusercontent.com/72430633/196582695-f0df78c4-65b7-4923-8383-0d1c619a6882.jpg)
![test精度可视化](https://user-images.githubusercontent.com/72430633/196582703-f5fb03f3-f368-4704-897c-5c930ed5eb21.jpg)
