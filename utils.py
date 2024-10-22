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
                img_set.append((img.astype(np.int16), label))
    else:
        raise FileNotFoundError    
    print(f"Loaded {len(img_set)} images")
    # print(type(img_set[0][0][0][0][0]))
    return np.array(img_set, dtype=object)

def plot_img(img_list):
    grid_size = int(np.ceil(np.sqrt(len(img_list))))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    if grid_size == 1:
        axs = np.array([axs])
        for i, image in enumerate(img_list):
            row = i // grid_size
            # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            axs[row].imshow(image)
            axs[row].axis('off')  # Turn off axes

    else:
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

def sampling(img_set, x):

    random_matrix = np.random.rand(len(img_set))
    R_sample = (x >= random_matrix).astype(int)
    return R_sample
    # R_sample = np.zeros(len(x))
    # x_10 = np.argpartition(x, -3)[-3:]
    # R_sample[x_10] = 1
    
    # subset = []
    # for i, e in enumerate(img_set):
    #     if R_sample[i] == 1:
    #         subset.append(e)
    # return np.array([subset])

def plot_data(img_set):
    labels = np.zeros(10)
    for e in img_set:
        labels[int(e[1])] += 1
    fig, ax = plt.subplots()
    ax.bar([0,1,2,3,4,5,6,7,8,9] , labels)
    plt.show()

if __name__ == "__main__":
    img_set = read_folder("./data/20/cifar10/5_clients/label_quantity_2/client_1", "cifar10")
    plot_data(img_set)
    
    # sampling("./data/mnist/dirichlet/client_1_new", "./results/client_2.txt")

