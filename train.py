from dataset import *
from model import *
import logging

#日志文件参数设置,日志输出到控制台
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                    filename="TF_log.log",
                    datefmt='%b %d %H:%M')

#命令行运行参数设置
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_iter', 500001, 'Total training iter')
flags.DEFINE_integer('validation_step', 100064, 'Total training iter')
flags.DEFINE_integer('step', 1000, 'Save after ... iteration')

#设置placeholder
with tf.name_scope("in"):
    left = tf.placeholder(tf.float32, [None, 72, 72, 3], name='left')
    right = tf.placeholder(tf.float32, [None, 72, 72, 3], name='right')
with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
    label = tf.to_float(label)

#两张图片分别经过siamesenet网络，得到两个向量
left_output = SIAMESE().siamesenet(left, reuse=False)
print(left_output.shape)

right_output = SIAMESE().siamesenet(right, reuse=True)

# predictions, loss, accuracy = SIAMESE().contrastive_loss(left_output, right_output, label)
model1, model2, distance, loss = SIAMESE().contrastive_loss(left_output, right_output, label)

global_step = tf.Variable(0, trainable=False)

# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.summary.scalar('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

# train_step = tf.train.MomentumOptimizer(0.0001, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)
# train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step=global_step)
train_step = tf.train.AdamOptimizer(0.00005).minimize(loss, global_step=global_step)  #优化损失

# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    # saver.restore(sess, 'checkpoint_trained/model_130000.ckpt')

    # setup tensorboard
    #标量数据汇总及记录
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    #变量直方图
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    #将所有summary全部保存到磁盘，以便tensorboard显示
    merged = tf.summary.merge_all()
    #指定一个文件用来保存图,可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中
    writer = tf.summary.FileWriter('train.log', sess.graph)

    left_dev_arr, right_dev_arr, similar_dev_arr = get_batch_image_array(left_dev, right_dev, similar_dev)

    # train iter
    print("train begin")
    idx = 0
    iter_count=0
    for i in range(FLAGS.train_iter):
        batch_left, batch_right, batch_similar, idx = get_batch_image_path(left_train, right_train, similar_train, idx)
        batch_left_arr, batch_right_arr, batch_similar_arr = \
            get_batch_image_array(batch_left, batch_right, batch_similar)

        _, l, summary_str = sess.run([train_step, loss, merged],
                                     feed_dict={left: batch_left_arr, right: batch_right_arr, label: batch_similar_arr})

        iter_count+=1
        if iter_count%10==0:
            print("current_iter: ",iter_count)

        #将训练过程数据保存在filewriter指定的文件中
        writer.add_summary(summary_str, i)
        print("\r#%d - Loss" % i, l)

        #validation
        if (i + 1) % FLAGS.validation_step == 0:
            DEV_NUMBER*=-1
            for k in range(64,DEV_NUMBER,64):
                val_distance = sess.run([distance],
                                        feed_dict={left: left_dev_arr[k-64:k], right: right_dev_arr[k-64:k], label: similar_dev_arr[k-64:k]})
                logging.info(np.average(val_distance))

        if i % FLAGS.step == 0 and i != 0:
            saver.save(sess, "checkpoint/model_%d.ckpt" % i)
