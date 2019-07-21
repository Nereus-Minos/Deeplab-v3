import tensorflow as tf
import numpy as np

# import time
import os

import matplotlib.pyplot as plt

# 具体导入请看deeplab_lfov下的__init__.py
# from .model import DeepLabLFOVModel搭建模型
# from .image_reader import ImageReader
# from .utils import decode_labels将灰度图转换为彩色图
# from .utils import inv_preprocess将BGR--->RGB
from deeplab_v3_model import DeepLabV3Model, ImageReader, decode_labels, inv_preprocess


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# 定义获取命令行参数的名字
FLAGS = tf.app.flags.FLAGS

# 定义命令行参数
tf.app.flags.DEFINE_integer("num_classes", 21, "数据集样本分类种类")
tf.app.flags.DEFINE_integer("batch_size", 4, "一批次数目")
tf.app.flags.DEFINE_string("data_dir", "./data/VOC2012/", "数据集目录")
tf.app.flags.DEFINE_string("data_list", "./data/train.txt", "训练集图片名称（方便图片读取）")
tf.app.flags.DEFINE_string("input_size", "321,321", "图片的尺寸（height,width）")
tf.app.flags.DEFINE_integer("num_steps", 200000, "迭代次数")
tf.app.flags.DEFINE_float("learning_rate", 2.5e-4, "学习率")
tf.app.flags.DEFINE_float("power", 0.9, "自适应学习率参数")
tf.app.flags.DEFINE_float("momentum", 0.9, "优化函数MomentumOptimizer的动量参数")
# tf.app.flags.DEFINE_string("restore_from", "./init_restore_model/deeplab_resnet.ckpt", "restore(预加载)模型参数目录")
tf.app.flags.DEFINE_string("restore_from", "./init_restore_model/", "restore(预加载)模型参数目录")
tf.app.flags.DEFINE_string("save_image_dir", "./images/train_save/", "保存预测结果分析图片目录")
tf.app.flags.DEFINE_integer("save_num_images", 2, "保存几张效果图")
tf.app.flags.DEFINE_integer("save_pred_every", 200, "循环多少次就保存一次真实图和效果图")
tf.app.flags.DEFINE_string("save_model_dir", "./save_model/", "保存训练模型参数的目录")
tf.app.flags.DEFINE_string("tensorboard_dir", "./tensorboard/", "存储tensorboard的目录")
tf.app.flags.DEFINE_float("weight_decay", 0.0005, "L2-loss正则化参数")


def main(argv=None):

    # map()会根据提供的函数对指定序列做映射。FLAGS.input_size.split(',')得到['321','321']
    image_height, image_width = map(int, FLAGS.input_size.split(','))
    input_size = (image_height, image_width)

    # 1.建立数据占位符或提供数据,tf.name_scope和tf.variable类似

    # 创建一个线程管理器
    coord = tf.train.Coordinator()

    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            FLAGS.data_dir,
            FLAGS.data_list,
            input_size,
            True,
            coord)
        image_batch, label_batch = reader.dequeue(FLAGS.batch_size)

    # 2.建立模型
    net = DeepLabV3Model(is_training=True, num_classes=FLAGS.num_classes)
    # If is_training=True, the statistics will be updated during the training.(BN)
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # 3.loss
    # 新加 L2正则损失
    pred, loss = net.loss(image_batch, label_batch, FLAGS.weight_decay)

    # 4.优化
    # 自适应学习率 = 学习率×（ 1 - iter/max_iter ）^power
    base_lr = tf.constant(FLAGS.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())    # any shape
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / FLAGS.num_steps), FLAGS.power))

    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc1' in v.name]     #
    conv_trainable = [v for v in all_trainable if 'fc1' not in v.name]   # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]   # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]    # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

    opt_conv = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, FLAGS.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, FLAGS.momentum)

    grads = tf.gradients(loss, conv_trainable + fc_w_trainable + fc_b_trainable)    # tf.gradients(ys,xs)实现ys对xs求导
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

    # 5.预测结果，得到的是灰度图
    pred = tf.expand_dims(pred, dim=3)  # Create 4D-tensor.

    # 6.在tensorboard中画图
    # Image summary.
    # tf.py_func
    images_summary = tf.py_func(inv_preprocess, [image_batch, FLAGS.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, FLAGS.save_num_images, FLAGS.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, FLAGS.save_num_images, FLAGS.num_classes], tf.uint8)

    total_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                     max_outputs=FLAGS.save_num_images)  # Concatenate row-wise.

    # loss
    loss_summary = tf.summary.scalar('loss', loss)

    # 7.创建保存模型的saver
    saver = tf.train.Saver(var_list=tf.global_variables())

    # 8.定义一个初始化变量op
    init_variable = tf.global_variables_initializer()

    # 9.会话，训练
    with tf.Session() as sess:
        # 9.1初始化所有变量
        sess.run(init_variable)

        # 9.2定义存储tensorboard的目录
        tensorboard_writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, graph=sess.graph)

        # 9.3如果有预加载模型（即fine-tune）(不预加载FC层)
        print("加载模型")
        # restore_var = [v for v in tf.global_variables() if 'fc1' not in v.name and 'Momentum' not in v.name]
        restore_var = [v for v in tf.global_variables()]
        ckpt = tf.train.get_checkpoint_state(FLAGS.restore_from)
        if ckpt and ckpt.model_checkpoint_path:
            print(restore_var)
            loader = tf.train.Saver(var_list=restore_var)
            loader.restore(sess, ckpt.model_checkpoint_path)
            print("Restored model parameters from {}".format(FLAGS.restore_from))
        print('ok')

        # if FLAGS.restore_from is not None:
        #     print(restore_var)
        #     loader = tf.train.Saver(var_list=restore_var)
        #     loader.restore(sess, FLAGS.restore_from)
        #     print("Restored model parameters from {}".format(FLAGS.restore_from))

        # 9.4开启读取数据的线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # 9.5迭代训练
        for step in range(FLAGS.num_steps):
            # start_time = time.time()

            feed_dict = {step_ph: step}     # 自适应学习率需要迭代步数

            # loss_value, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            _ = sess.run(train_op, feed_dict=feed_dict)

            # 保存预测效果图和模型参数
            if step % FLAGS.save_pred_every == 0:

                # 保存效果对比图
                images, labels, preds, summary = sess.run([image_batch, label_batch, pred, total_summary])

                tensorboard_writer.add_summary(summary, step)

                fig, axes = plt.subplots(FLAGS.save_num_images, 3, figsize=(16, 12))
                for i in range(FLAGS.save_num_images):
                    axes.flat[i * 3].set_title('data')
                    axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                    axes.flat[i * 3 + 1].set_title('mask')

                    # labels[i, :, :, 0]得到的是一副shape为[height,width]的图片
                    axes.flat[i * 3 + 1].imshow(decode_labels(labels, num_images=FLAGS.batch_size)[i])

                    axes.flat[i * 3 + 2].set_title('pred')
                    axes.flat[i * 3 + 2].imshow(decode_labels(preds, num_images=FLAGS.batch_size)[i])
                plt.savefig(FLAGS.save_image_dir + str(step) + ".png")
                plt.close(fig)

                # 保存模型
                model_name = 'model.ckpt'
                checkpoint_path = os.path.join(FLAGS.save_model_dir, model_name)
                saver.save(sess, checkpoint_path, global_step=step)

                # mIoU
                IoU = 0.0

                with tf.name_scope("mIoU"):

                    for i in range(FLAGS.batch_size):

                        val_predictions = tf.reshape(pred[i], [-1, ])
                        val_labels = tf.reshape(tf.cast(label_batch[i], tf.int32), [-1, ])

                        val_mask = val_labels <= FLAGS.num_classes - 1

                        val_predictions = tf.boolean_mask(val_predictions, val_mask)
                        val_labels = tf.boolean_mask(val_labels, val_mask)

                        # Define the evaluation metric.
                        mIoU_, update_op = tf.contrib.metrics.streaming_mean_iou(val_predictions, val_labels,
                                                                                FLAGS.num_classes)

                        # mIoU 一定要有
                        sess.run(tf.local_variables_initializer())

                        _ = sess.run(update_op)

                        IoU += mIoU_.eval(session=sess)

                    mIoU = IoU / FLAGS.batch_size

                    print('Mean IoU: {:.3f}'.format(mIoU))

                    # mIoU
                    mIoU_summary = tf.summary.scalar('mIoU', tf.convert_to_tensor(mIoU))

                add_loss_summary, add_mIoU_summary = sess.run([loss_summary, mIoU_summary])

                tensorboard_writer.add_summary(add_loss_summary, step)
                tensorboard_writer.add_summary(add_mIoU_summary, step)


            # duration = time.time() - start_time

            # print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

        # 9.6关闭线程
        coord.request_stop()
        coord.join(threads)

    return None


if __name__ == "__main__":
    tf.app.run()
