import os
import argparse
import cv2 as cv
import numpy as np

from submod import submod_maximize
from similarity import cal_sim_matrix
from utils import read_folder, plot_img

DATA_FOLDER = "./data/CIFAR10/label_quantity_3"
RESULT_FOLDER = "./results"
NUM_CLIENT = 2

NUM_SAMPLES = 100
CONSTRAINT = 10
LEARNING_RATE = 0.05
ITER = 1000

def client_work(in_folder, out_folder, ci, lr, k):
    # print(f"Clien: {i+1}, pid: {os.getpid()}")
    print(f"\nClient {ci+1} started!")
    dataset = in_folder.split("/")[1]

    img_set = read_folder(in_folder, dataset_name=dataset)
    sim_matrix = cal_sim_matrix(img_set, ci, out_folder)
    x, f_A, A_index = submod_maximize(len(img_set), eta=lr, max_iter=ITER, c=k, matrix = sim_matrix)
    
    np.savetxt(os.path.join(out_folder, f"client_{ci+1}.txt"),x, delimiter = ",")

    random_array = np.random.rand(len(img_set))
    R_index = (x >= random_array).astype(int)

    selected = []
    for i, img in enumerate(img_set):
        if R_index[i]:
            selected.append(img) 
    
    print(f"Client {i+1} finished!")

    return selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distrubuted Submodulr setting")

    parser.add_argument("-k", "--constraint", type=int, required=True)
    parser.add_argument("-lr", "--learningrate", type=float, default=0.1)
    parser.add_argument("-in", "--input_folder", type=str, required=True)
    parser.add_argument("-out", "--output_folder", type=str, required=True)
    parser.add_argument("-d", "--directory", type=str, required=True)

    args = parser.parse_args()

    num_clis = NUM_CLIENT

    data_folder = os.path.join(args.input_folder, args.directory)
    result_folder = os.path.join(args.output_folder, args.directory)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    learning_rate = args.learningrate
    constraint = args.constraint

    print(f"\n======================================")
    print(f"STARTING: {args.directory}\n - Number of clients: 5;\n - Learning rate: {args.learningrate};\n - Constraint k = {args.constraint}; ")

    B = []
    A_list = []
    fA_list = []

    for i in range(num_clis):
        A, f_A = client_work(os.path.join(data_folder, f"client_{i}"), result_folder, i, learning_rate, constraint)
        A_list.append(A)
        fA_list.append(f_A)
        for j in range(len(A)):
            B.append(A[j])
    b_imgs = [b[0] for b in B]
    plot_img(b_imgs)
    print("Aggrerating clients' solutions ...")
    B_matrix = cal_sim_matrix(B, num_clis+1 , result_folder)

    x_Agc, f_Agc, Agc_index = submod_maximize(len(B), eta=learning_rate, max_iter=ITER, c=constraint, matrix = B_matrix)

    Agc = []
    for i, img in enumerate(B):
        if Agc_index[i] == 1:
            Agc.append(img) 

    r = np.savetxt(os.path.join(result_folder, f"x_Agc.txt"),x_Agc, delimiter = ",")

    A_list.append(Agc)
    fA_list.append(f_Agc)
    r = np.savetxt(os.path.join(result_folder, f"f_A.txt"),np.array(fA_list), delimiter = ",")

    print(f"Global solution f_A: {max(fA_list)}")

    max = A_list[np.argmax(np.array(fA_list))]    
    max_imgs = [m[0] for m in max]

    for i, img in enumerate(max_imgs):
        cv.imwrite(os.path.join(result_folder, f"result_{i}.jpg"), img)

    # plot_img(max_imgs)
    # processes=[]
    # output_queue = mp.Queue()
    # for i, folder in enumerate(folder_list):
    #     process = mp.Process( target=client_work, args=(folder, i,), name=f"Process-{i+1}" )
    
    #     processes.append(process)
    #     process.start()
    #     print(f"Client {i+1} started on process {process.name}")

    # for process in processes:
    #     process.join()
    #     print(f"{process.name} completed")

    # results = []
    # while not output_queue.empty():
    #     results.append(output_queue.get())

    
