import os
import numpy as np
from tqdm import tqdm

# from data import download_data
def set2array(img_set: set):
    img_array = np.zeros(shape=(len(img_set), 32, 32, 3), dtype=np.uint8) # create a 4D array to store the images
    for i, e in tqdm(enumerate(img_set),desc = "Convert to array"):
        img_array[i] = np.array(e[1], dtype=np.uint8).reshape((32, 32, 3)) #convert python set to array
    return img_array

def cal_sim(img_set1, img_set2, folder=None):
	img_shape = img_set1[0][0].shape
	matrix = np.zeros(shape=(len(img_set1),len(img_set2)))
	for i in range(len(img_set1)):
		for j in range(len(img_set2)):
			if len(img_shape) == 3:
				diff = np.sum(np.absolute(img_set1[i][0] - img_set2[j][0])) / (img_shape[0]*img_shape[1]*img_shape[2]) /255  # normalize
			elif len(img_shape) == 2:
				diff = np.sum(np.absolute(img_set1[i][0] - img_set2[j][0])) / (img_shape[0]*img_shape[1]) /255  # normalize
			similarity = 1 - diff
			matrix[i][j] = similarity
	if folder:
		r = np.savetxt(os.path.join(folder, f"sim.csv"), matrix, delimiter=',')
	return matrix

def cal_sim_matrix(img_set, client_i, folder):
	size = len(img_set)
	img_shape = img_set[0][0].shape

	upper_half = np.zeros(shape=(size, size))

	print("Calculating Similarity ...")
	for i in(range(size)):
		for j in range(size):
			if i < j:
				if len(img_shape) == 3:
					diff = np.sum(np.absolute(img_set[i][0] - img_set[j][0]))/ (img_shape[0]*img_shape[1]*img_shape[2]) /255  # normalize
				elif len(img_shape) == 2:
					diff = np.sum(np.absolute(img_set[i][0] - img_set[j][0])) / (img_shape[0]*img_shape[1]) /255  # normalize
				similarity = 1 - diff
				upper_half[i][j] = similarity #calculate matrix's upper halft

	sim_matrix = upper_half + upper_half.T
	if client_i == -1:
		r = np.savetxt(os.path.join(folder, f"{client_i}.txt"), sim_matrix, delimiter=',')
	else:
		r = np.savetxt(os.path.join(folder, f"client_{client_i}.txt"), sim_matrix, delimiter=',')
	return sim_matrix

if __name__ == "__main__":
	matrix = cal_sim_matrix()
