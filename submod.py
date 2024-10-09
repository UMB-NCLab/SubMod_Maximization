import os
import argparse
import cvxpy as cp
import numpy as np
from utils import read_folder, plot_img
from similarity import cal_sim_matrix

NUM_SAMPLES = 50
CONSTRAINT = 10
LEARNING_RATE = 0.05
ITER = 1000

#Facility location
def f(R, matrix, f_dict):
    R_tuple = tuple(R)
    if R_tuple not in f_dict:
        selected_indices = np.where(R == 1)[0]
        # print(f"Sampled: {selected_indices}")
        if selected_indices.size == 0:
            max_similarity = 0
        else:
            max_similarities = matrix[selected_indices, :].max(axis=0)
            # print(max_similarities)
            max_similarity = max_similarities.sum()
            # print(max_similarity)
        f_dict[R_tuple] = max_similarity/len(matrix[0])
    return f_dict[R_tuple]

def compute_gradient_sample(R, E_size, matrix, f_dict):
    grad = np.zeros(E_size)
    for i in range(E_size):
        Rd = R.copy()
        Rd[i] = 0
        Ru = R.copy()
        Ru[i] = 1
        grad[i] = f(Ru, matrix, f_dict) - f(Rd, matrix, f_dict)
    return grad

def gradient_step(E_size, x, num_sample, matrix, f_dict):
    grad_f = np.zeros(E_size)
    
    random_matrix = np.random.rand(num_sample, E_size)
    R_matrix = (x >= random_matrix).astype(int)

    print("Conputing Gradient ...")
    for n in range(num_sample):
        grad_f_n = compute_gradient_sample(R_matrix[n], E_size, matrix, f_dict)
        grad_f +=  grad_f_n  

        # results = Parallel(n_jobs=-1)(
        #     delayed(compute_gradient_sample)(R_matrix[n], E_size, matrix) for n in range(num_sample))
        # for grad in results:
        #     grad_f += grad
    
    grad_f /= num_sample
    return grad_f

def submod_maximize(E_size: int, eta: float, max_iter: int, k: int, matrix: np.ndarray):
    x = np.zeros(E_size)
    i = 0
    num_sample = NUM_SAMPLES
    f_dict = {}
    # while (i < max_iter) and np.all(x < 1):
    while (i < max_iter) and np.all(x < 1):
        grad_f = gradient_step(E_size, x, num_sample, matrix, f_dict) 
        
        print(f"Iteration {i+1}, Grad_f: {grad_f[:10]}")
        # print(f"\nIteration {i+1}:\n Grad_f: {grad_f}")
        w = cp.Variable(len(x))
        
        constraints = [
            w >= 0,
            w <= 1,
            cp.sum(w) <= k
        ]
        objective = cp.Maximize(w @ grad_f)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if w.value is None:
            print("Optimization failed.")
            break
        
        v = w.value
        x += eta * v
        
        # Optional: Project x to stay within [0,1]
        x = np.clip(x, 0, 1)
        
        # print(f"\nV: {v[:10]}")
        # print(f"\nX: {x[:10]}")
        # print(f"\nV: {v}")
        # print(f"\nX: {x}")    
        A_index = np.zeros(E_size)
        # print(f"argpart: {np.argpartition(x, -c)}")
        x_10 = np.argpartition(x, -k)[-k:]
        A_index[x_10] = 1
        f_A = f(A_index, matrix, f_dict)

        print(f"Selected: {x_10}")
        print(f"f_S: {f_A}")

        f_dict.clear()
        i += 1  # Correct increment 

    return x, f_A, A_index

# DATA_FOLDER = ".\data\MNIST\IID\client_0"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Distrubuted Submodulr setting")

    parser.add_argument("-k", "--constraint", type=int, required=True)
    parser.add_argument("-lr", "--learningrate", type=float, default=0.1)
    parser.add_argument("-in", "--input_folder", type=str, required=True)
    parser.add_argument("-out", "--output_folder", type=str, required=True)
    parser.add_argument("-d", "--directory", type=str, required=True)

    args = parser.parse_args()

    data_folder = os.path.join(args.input_folder, args.directory)
    result_folder = os.path.join(args.output_folder, args.directory)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    learning_rate = args.learningrate
    constraint = args.constraint

    print(f"\nCentralized started!")
    dataset = data_folder.split("/")[1]

    img_set = read_folder(data_folder, dataset_name=dataset)
    sim_matrix = cal_sim_matrix(img_set, -1, result_folder)
    x, f_A, A_index = submod_maximize(len(img_set), eta=learning_rate, max_iter=ITER, c=constraint, matrix = sim_matrix)
    
    np.savetxt(os.path.join(result_folder, f"centralized.txt"),x, delimiter = ",")

    # random_array = np.random.rand(len(img_set))
    # R_index = (x >= random_array).astype(int)

    A = []
    for i, img in enumerate(img_set):
        if A_index[i] == 1:
            A.append(img) 
    
    print(f"Centralized finished!")
