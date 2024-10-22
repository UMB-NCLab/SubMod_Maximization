import os
import time
import argparse
import cv2 as cv
import numpy as np

from submod import submod_maximize, f
from similarity import cal_sim_matrix, cal_sim
from utils import read_folder, plot_img, sampling
from dclient import Client

DATA_FOLDER = "./data/CIFAR10/label_quantity_3"
RESULT_FOLDER = "./results"
NUM_CLIENT = 5

NUM_SAMPLES = 100
CONSTRAINT = 10
LEARNING_RATE = 0.05
ITER = 100

def topology(num_clis, type, k = 0):
    topology = np.zeros((num_clis, num_clis))
    #Exponential topology
    if type == "exp":
        for i in range(num_clis):
            topology[i][(i + 1 + k)%num_clis] = 1 #k = 0 => Ring topology

    #Ring topology
    elif type == "ring":
        for i in range(num_clis):
            topology[i][(i+1)%num_clis] = 1 
    
    #Star topology
    elif type == "star":
        for i in range(num_clis):
            if i != k:
                # topology[k][i] = 1
                topology[i][k] = 1
    
    #Grid topology
    elif type == "grid":
        # 1 _ 2 _ 3
        # |   |
        # 4   5
        topology = np.ndarray([[0,1,0,1,0],
                               [1,0,1,0,1],
                               [0,1,0,0,1],
                               [1,0,0,0,1],
                               [0,1,1,1,0]])

    #Random topology
    elif type == "random":
        rand = np.zeros(num_clis * num_clis, dtype=int)
        indices = np.random.choice(range(num_clis * num_clis), size=num_clis, replace=False)
        rand[indices] = 1        
        topology = rand.reshape((num_clis, num_clis))
    return topology.astype(int)

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Distrubuted Submodulr setting")

    parser.add_argument("-k", "--constraint", type=int, required=True)
    parser.add_argument("-T", "--rounds", type=float, default=0.1)
    parser.add_argument("-in", "--input_folder", type=str, required=True)
    parser.add_argument("-out", "--output_folder", type=str, required=True)
    parser.add_argument("-d", "--directory", type=str, required=True)
    parser.add_argument("-t", "--topology", type=str, required=True)
    args = parser.parse_args()

    num_clis = NUM_CLIENT

    data_folder = os.path.join(args.input_folder, args.directory)
    result_folder = os.path.join(args.output_folder, args.directory)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    rounds = int(args.rounds)
    k = args.constraint
    topo_type = args.topology
    dataset = args.directory.split("/")[0]

    with open(f'f_value_{topo_type}.txt', 'w+') as fp:
        pass  

    print(f"\n======================================")
    print(f"STARTING: {args.directory}\n - Number of clients: 5;\n - Learning rate: {rounds};\n - Constraint k = {k}; ")

    print("Reading E set...")
    E = read_folder(os.path.join(args.input_folder,dataset,"centralized"), dataset_name=dataset)

    clients = []
    for i in range(num_clis):
        print(f"Client {i+1} reading V_{i+1} set...")
        V_i = read_folder(os.path.join(data_folder, f"client_{i}"), dataset_name=dataset)
        matrix = cal_sim(E, V_i)
        client = Client(index= i, 
                        T = rounds,
                        sim_matrix=matrix)
        clients.append(client)

    

    print("\nStart maximizing submodular...")
    for t in range(rounds):
        print(f"\nRound {t+1}:")

        _topology = topology(num_clis, type=topo_type, k = (t % (num_clis-1)))
        print(f"Topology: {topo_type}")
        print(_topology)
        for c in clients:
            c.get_neighbors(_topology)

        for c in clients:
            print(f"\nClient {c.index+1}:")    
            x,d = c.action(T=rounds) 
            print(f"X: {x[:10]}\nD: {d[:10]}")
        
        R_E = sampling(E, x)
        E_matrix = cal_sim(E, E)
        f_dict = {}
        f_value = f(R_E, E_matrix, f_dict)    
        f_dict.clear()
        print(f"Round {t+1}'s f value: {f_value}.")
        with open(f'f_value_{topo_type}.txt', 'a') as fp:
            np.savetxt(fp, np.array([f_value]), fmt='%f')

    print("Calculating final value...")
    R_E = sampling(E, x)
    E_matrix = cal_sim(E, E)
    f_dict = {}
    f_value = f(R_E, E_matrix, f_dict)    
    f_dict.clear()

    print(f"Final f value: {f_value}.")
    with open(f'final.txt', 'a') as fp:
        fp.write(f"{topo_type}: ")
        np.savetxt(fp, np.array([f_value]), fmt='%f')

    max_imgs = [e[0] for i, e in enumerate(E) if R_E[i] == 1]
    for i, img in enumerate(max_imgs):
        cv.imwrite(os.path.join(result_folder, f"result_{i}.jpg"), img)

    print(f"Total time executed: {time.time() - start}")

    
