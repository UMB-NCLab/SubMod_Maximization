import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List

#Read folder return image 
def read_folder(folder, dataset_name):
    if(os.path.exists(folder)):
        img_set = []
        ls = os.listdir(folder)
        ls.sort()
        # print(ls[:10])
        for file in ls:
            if(file.endswith(('.png', '.jpg', '.jpeg'))):
                label = file.split("_")[0]
                if dataset_name == "mnist":
                    img = cv.imread(os.path.join(folder, file), cv.IMREAD_GRAYSCALE)
                else:
                    img = cv.imread(os.path.join(folder, file))
                img_set.append((img, label))
    else:
        raise FileNotFoundError    
    print(f"Loaded {len(img_set)} images")
    return np.array(img_set,dtype=object)

def plot_img(img_list, label_list):
    grid_size = int(np.ceil(np.sqrt(len(img_list))))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, image in enumerate(img_list):
        row = i // grid_size
        col = i % grid_size
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        axs[row, col].imshow(image)
        axs[row, col].axis('off')  # Turn off axes

    for j in range(i + 1, grid_size * grid_size):
        row = j // grid_size
        col = j % grid_size
        axs[row, col].axis('off')

    # Display the grid
    plt.tight_layout()
    plt.show()

def sampling(folder, x_file) -> Tuple[List, List]:

    img_set = read_folder(folder=folder, dataset_name="mnist")

    x = np.float32(np.loadtxt(x_file, delimiter=","))

    random_matrix = np.random.rand(len(img_set))
    R_sample = (x >= random_matrix).astype(int)
    
    img_list = []
    label_list = []
    for i, img in enumerate(img_set):
        if R_sample[i] > 0:
            img_list.append(img[0])
            label_list.append(img[1])
    
    return img_list, label_list
            
if __name__ == "__main__":
    img_list,_ = sampling("./data/mnist/dirichlet/client_0_new", "./results/client_1.txt")
    plot_img(img_list)
    
    # sampling("./data/mnist/dirichlet/client_1_new", "./results/client_2.txt")

