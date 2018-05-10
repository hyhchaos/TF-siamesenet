import tensorflow as tf
import random
from dataset import *

#取三千个做accuary检测
file_positive = open('E:\\faceTF\\positive_pairs_path.txt', 'r')
file_negative = open('E:\\faceTF\\negative_pairs_path.txt', 'r')

images_positive = [line.strip() for line in file_positive.readlines()]
images_negative = [line.strip() for line in file_negative.readlines()]

images_positive=random.sample(images_positive,3000)
images_negative=random.sample(images_negative,3000)

images = np.asarray(images_positive + images_negative)
labels = np.append(np.ones([3000]), np.zeros([3000]))

np.random.seed(3)
shuffle_indices = np.random.permutation(np.arange(len(labels)))
images_shuffled = images[shuffle_indices]
labels_shuffled = labels[shuffle_indices]


total_accuary=0
total_count=0

graph = tf.Graph()
with graph.as_default():
    #尽量用gpu跑
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #加载模型
        saver = tf.train.import_meta_graph('checkpoint/model_104000.ckpt.meta')
        saver.restore(sess, 'checkpoint/model_104000.ckpt')

        left = graph.get_operation_by_name("in/left").outputs[0]
        right = graph.get_operation_by_name("in/right").outputs[0]

        distance = graph.get_operation_by_name("output/distance").outputs[0]

        image_test = []
        label_test = []
        index = 1

        # Generate batches for one epoch
        for image, label in zip(images_shuffled, labels_shuffled):
            index += 1
            image_test.append(image)
            label_test.append(label)
            if index % 100 == 0 and index > 0:
                left_test = []
                right_test = []
                for image_one in image_test:
                    line_one_list = str(image_one).split(' ')
                    left_test.append(line_one_list[0])
                    right_test.append(line_one_list[1])
                left_test_arr, right_test_arr, _ = get_batch_image_array(left_test, right_test, [])

                output_distance = sess.run([distance], feed_dict={left: left_test_arr, right: right_test_arr})
                output_distance = output_distance[0]

                true_num = 0
                for distance_one, label_one,left_one,right_one in zip(output_distance, label_test,left_test,right_test):
                    print("distance: ",distance_one," label: ",label_one," left_img: ",left_one," right_img: ",right_one)
                    if float(distance_one) < 0.5:
                        same_flag = 0
                    else:
                        same_flag = 1
                    if label_one == same_flag:
                        true_num += 1

                print("accuary: ",true_num / 100.)
                total_accuary+=true_num/100
                total_count+=1

                image_test = []
                label_test = []
        print("total count: ",total_count)
        print("total_accuary: ",total_accuary/total_count)