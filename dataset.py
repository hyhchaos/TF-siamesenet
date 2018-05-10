import numpy as np
from PIL import Image
import os

# BASE_PATH = 'same_size/lfw/'
BASE_PATH = 'E:/images/'
DEV_NUMBER = -2000
batch_size = 64

#获取数据
negative_pairs_path_file = open('E:\\faceTF\\negative_pairs_path.txt', 'r')
negative_pairs_path_lines = negative_pairs_path_file.readlines()
positive_pairs_path_file = open('E:\\faceTF\\positive_pairs_path.txt', 'r')
positive_pairs_path_lines = positive_pairs_path_file.readlines()

#主要分成三类
left_image_path_list = []
right_image_path_list = []
similar_list = []

for line in negative_pairs_path_lines:
    left_right = line.strip().split(' ')        #移除头尾空格并分隔开每一项
    #两张不同照片
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    #0表示不相似
    similar_list.append(0)

for line in positive_pairs_path_lines:
    left_right = line.strip().split(' ')
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    similar_list.append(1)

#从列表类型转化成array类型
left_image_path_list = np.asarray(left_image_path_list)
right_image_path_list = np.asarray(right_image_path_list)
similar_list = np.asarray(similar_list)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(similar_list)))   #随机数组
left_shuffled = left_image_path_list[shuffle_indices]
right_shuffled = right_image_path_list[shuffle_indices]
similar_shuffled = similar_list[shuffle_indices]

# print(left_shuffled[:5])
# print(right_shuffled[:5])
# print(similar_shuffled[:5])

# Split train/test set
left_train, left_dev = left_shuffled[:DEV_NUMBER], left_shuffled[DEV_NUMBER:]
right_train, right_dev = right_shuffled[:DEV_NUMBER], right_shuffled[DEV_NUMBER:]
similar_train, similar_dev = similar_shuffled[:DEV_NUMBER], similar_shuffled[DEV_NUMBER:]


# print(left_train[:5])
# print(right_train[:5])
# print(similar_train[:5])


#根据图片地址获取图片并转化为array类型，返回image的list
def vectorize_imgs(img_path_list):
    image_arr_list = []
    for img_path in img_path_list:
        # print("path: ", BASE_PATH + img_path)
        if os.path.exists(BASE_PATH + img_path):
            img = Image.open(BASE_PATH + img_path)
            img=img.resize((72,72),Image.BILINEAR)
            img_arr = np.asarray(img, dtype="float32")
            # print (img_arr.shape)
            image_arr_list.append(img_arr)
        else:
            print("not exist image: ",img_path)
    return image_arr_list

#
def get_batch_image_path(left_train, right_train, similar_train, start):
    end = (start + batch_size) % len(similar_train)
    if start < end:
        return left_train[start:end], right_train[start:end], similar_train[start:end], end
    # 当 start > end 时，从头返回
    return np.concatenate([left_train[start:], left_train[:end]]), \
           np.concatenate([right_train[start:], right_train[:end]]), \
           np.concatenate([similar_train[start:], similar_train[:end]]), \
           end

#图片按batch转为array
def get_batch_image_array(batch_left, batch_right, batch_similar):
    return np.asarray(vectorize_imgs(batch_left),dtype="float32") / 255., \
           np.asarray(vectorize_imgs(batch_right),dtype="float32") / 255., \
           np.asarray(batch_similar)[:, np.newaxis]


if __name__ == '__main__':
    # pass
    idx = 0
    batch_left, batch_right, batch_similar, idx = get_batch_image_path(left_train, right_train, similar_train, idx)
    print("batch_left: ",batch_left[:5])
    print("batch_right: ",batch_right[:5])
    print("batch_similar: ",batch_similar[:5])
    print("get_batch_image_array: ",get_batch_image_array(batch_left[:5], batch_right[:5], batch_similar[:5]))
