import tensorflow as tf
import random
import cv2
import detect_face
from dataset import *


#get ref_img and tar_img
reference_img_path="E:\\faceTF\\TF-siamesenet\\my_img\\ref.PNG"
reference_img=Image.open(reference_img_path)
reference_img=reference_img.resize((72,72),Image.BILINEAR)
reference_img_arr = np.asarray(reference_img, dtype="float32")/255
reference_img_list=[]
reference_img_list.append(reference_img_arr)

target_img_path="E:\\faceTF\\TF-siamesenet\\my_img\\tar.PNG"
target_img=Image.open(target_img_path)
target_img=target_img.resize((72,72),Image.BILINEAR)
target_img_arr = np.asarray(target_img, dtype="float32")/255
target_img_list=[]
target_img_list.append(target_img_arr)

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

        output_distance = sess.run([distance], feed_dict={left: reference_img_list, right: target_img_list})
        output_distance = output_distance[0]

        print("distance: ", output_distance)
        if float(output_distance) < 0.5:
            print ("other")
        else:
            print("you")
