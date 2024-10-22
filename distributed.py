import os
import time
import argparse
import cv2 as cv
import numpy as np

import multiprocessing as mp
from submod import submod_maximize, f
from similarity import cal_sim_matrix, cal_sim
from utils import read_folder, plot_img

DATA_FOLDER = "./data/CIFAR10/label_quantity_3"
RESULT_FOLDER = "./results"
NUM_CLIENT = 5

NUM_SAMPLES = 100
CONSTRAINT = 10
LEARNING_RATE = 0.05
ITER = 100

def client_work(in_folder, out_folder, ground_set, ci, lr, k, mp_queue=None):
    # print(f"Clien: {i+1}, pid: {os.getpid()}")
    print(f"\nClient {ci+1} started!")
    start = time.time()

    dataset = in_folder.split("/")[1]
    print(f"Client {ci+1} reading V{ci+1} set...")
    img_set = read_folder(os.path.join(in_folder, f"client_{i}"), dataset_name=dataset)
    sim_matrix = cal_sim(img_set,ground_set)
    x, f_A, A_index = submod_maximize(len(img_set), eta=lr, max_iter=ITER, k=k, matrix = sim_matrix)
    
    np.savetxt(os.path.join(out_folder, f"client_{ci+1}.txt"),x, delimiter = ",")

    x_10 = np.argpartition(x, -k)[-k:]
    selected = img_set[x_10]
    # selected.append(img) 
    
    print(f"Client {ci+1} finished! Time executed:{time.time()-start}")

    # Concurrent setting
    if mp_queue:
        mp_queue.put((A_index, selected))
    
    #Sequential 
    else:
        return A_index, selected

if __name__ == "__main__":
    start = time.time()
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
    k = args.constraint
    dataset = args.directory.split("/")[0]

    print(f"\n======================================")
    print(f"STARTING: {args.directory}\n - Number of clients: 5;\n - Learning rate: {args.learningrate};\n - Constraint k = {args.constraint}; ")

    # q = mp.Queue()
    # processes=[]
    # for i in range (num_clis):
    #     process = mp.Process( target=client_work,args= (
    #                               os.path.join(data_folder, f"client_{i}"),
    #                               result_folder,
    #                               i,
    #                               learning_rate,
    #                               k, q),
    #                               name=f"Process-{i+1}" )
    
    #     processes.append(process)
    #     process.start()
    #     print(f"Client {i+1} started on process {process.name}")

    # for process in processes:
    #     process.join()
    #     print(f"{process.name} completed")

    # results = []
    # while not q.empty():
    #     results.append(q.get())

    B = []
    A_list = []
    fA_list = []
    index_list = []
    print("Server reading V set...")
    V = read_folder(os.path.join(args.input_folder,dataset,"centralized"), dataset_name=dataset)
    # for i, A_index, A in enumerate(results):
    #     A_list.append(A)
    #     index_list.append(A_index)
    #     for j in range(len(A)):
    #         B.append(A[j])
    # print(f"Got {len(B)} images.")
    for i in range(num_clis):
        A_index, A = client_work(data_folder, result_folder, V, i, learning_rate, k)
        A_list.append(A)
        # fA_list.append(f_A)
        for j in range(len(A)):
            B.append(A[j])
    print(f"Got {len(B)} images.")

    print("Aggrerating clients' solutions ...")
    # f_A, A_index, A
    # for ri, (index, A) in enumerate(results):
    #     index_list.append(index)
    #     A_list.append(A)
    #     for j in range(A):
    #         B.append(A[j])
    # print(len(B))

    print("Computing aggrerated set ...")
    print("Computing A_i sets ...")
    
    f_dict = {}
    for i, A_i in enumerate(A_list):
        # A_imgs = [a[0] for a in A_list]
        A_matrix = cal_sim(V, A_i)
        R_A = np.ones(k)
        fA_list.append(f(R_A, A_matrix, f_dict))
        print(fA_list[i])
        f_dict.clear()

    print("Computing B set ...")
    B_imgs = [b[0] for b in B]
    # plot_img(B_imgs)
    # B_matrix = cal_sim_matrix(B, num_clis+1 , result_folder)
    B_B_matrix = cal_sim(B, B)
    x_Agc, f_Agc, Agc_index = submod_maximize(len(B), eta=learning_rate, max_iter=ITER, k=k, matrix = B_B_matrix)
    
    fA_list.append(f_Agc)
    Agc = []
    x_10 = np.argpartition(x_Agc, -k)[-k:]
    for i in x_10:    
        Agc.append(B[i])
    A_list.append(Agc)
    fA_list.append(f_Agc)

    print(f"Global solution f_A: {max(fA_list)}")

    max = A_list[np.argmax(np.array(fA_list))]    
    max_imgs = [m[0] for m in max]

    r = np.savetxt(os.path.join(result_folder, f"x_Agc.txt"),x_Agc, delimiter = ",")

    r = np.savetxt(os.path.join(result_folder, f"f_A.txt"),np.array(fA_list), delimiter = ",")

    for i, img in enumerate(max_imgs):
        cv.imwrite(os.path.join(result_folder, f"result_{i}.jpg"), img)

    print(f"Total time executed: {time.time() - start}")

    
