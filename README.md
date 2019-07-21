deeplab-V3(使用aspp结构)改进：
    1、在ASPP中使用了BN层；
    2、ASPP（所有卷积核都为256个）变为一个1*1卷积、三个3*3空洞卷积（采样率分别为（6、12、18））；
    和一个图像级特征（即通过block4输出的特征做全局平均池化tf.reduce_mean(block4,[1,2])+1*1conv）；
    3、ASPP经过concat后通过一个1*1大小为256的conv，之后在经过1*1大小为num_classes的conv得到最终的特征；
    4、在计算loss时，现将pred做八倍上采样，再与truth label做交叉熵损失；
    5、没有使用CRF
