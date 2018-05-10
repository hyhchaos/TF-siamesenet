import os
import random
from itertools import combinations,product


clean_list_file=open("E:\\faceTF\\MS-Celeb-1M_clean_list.txt","r")
clean_list_lines=clean_list_file.readlines()

img_number=5049823
positive_pair_number=1500000
negative_pair_number=1500000
img_name_set_list=[]

def distribute_img_by_id():
	count = 0
	img_id = "0\n"
	img_location_set = set()
	for i in clean_list_lines:
		i_location,i_id=i.split(" ")
		if i_id==img_id:
			img_location_set.add(i_location)
		else:
			img_id=i_id
			img_name_set_list.append(img_location_set)
			img_location_set=set()
			img_location_set.add(i_location)
		count+=1
		if count==img_number:
			img_name_set_list.append(img_location_set)
			break


def generate_positive_text():
	positive_count=0
	positive_text_set=set()
	positive_out_set = []

	#generate positive pair
	for i in img_name_set_list:
		positive_text_set.add(combinations(i,2))

	print(len(positive_text_set))

	#randomly choose pairs
	positive_pairs_file = open("E:\\faceTF\\positive_pairs_path.txt", "w")
	for j in positive_text_set:
		positive_pair_list=list(j)
		for k in positive_pair_list:
			positive_count+=1
			positive_out_set.append((k[0],k[1]))
		if positive_count >= positive_pair_number:
			break
	# random.shuffle(positive_out_set)

	#write into target file
	for m in positive_out_set:
		positive_pairs_file.write(m[0] + " " + m[1] + "\n")
	print("positive_count: ",positive_count)


def generate_negative_text():
	negative_count=0
	negative_text_set=set()
	negative_out_set = []
	for i in range(10000):
		slice = random.sample(img_name_set_list, 2)
		negative_text_set.add(product(slice[0],slice[1]))

	print(len(negative_text_set))

	negative_pairs_file = open("E:\\faceTF\\negative_pairs_path.txt", "w")
	for j in negative_text_set:
		negative_pair_list=list(j)
		for k in negative_pair_list:
			negative_count+=1
			negative_out_set.append((k[0],k[1]))
		if negative_count >= negative_pair_number:
			break
	# random.shuffle(negative_out_set)
	for m in negative_out_set:
		negative_pairs_file.write(m[0] + " " + m[1] + "\n")
	print("negative_count: ",negative_count)





if __name__=='__main__':
	distribute_img_by_id()
	generate_positive_text()
	generate_negative_text()