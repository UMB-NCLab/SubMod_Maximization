from submod import argmax_v, gradient_estimate
import numpy as np

K = 10
ALPHA = 0.5
NUM_SAMPLE = 50

class Client:
    
    all_clients = []

    def __init__(self, index, T, sim_matrix):
        self.index = index
        self.x = np.zeros(len(sim_matrix[:,0]))
        self.d = np.zeros(len(sim_matrix[:,0]))
        self.alpha = ALPHA
        self.T = T
        self.matrix = sim_matrix
        self.neighbor_list = []
        self.weight = 0
        self.f_dict = {}
        Client.all_clients.append(self)
        
    def get_neighbors(self, topology):
        self.neighbor_list = []
        index = self.index
        for i in range(len(topology[:, index])):
            if topology[i][index] == 1:  
                self.neighbor_list.append(self.get_client(i))
        print(f"Client {index+1} has {len(self.neighbor_list)} neighbor(s).")
        # print(self.neighbor_list)
        self.weight = 1/(len(self.neighbor_list)+1)

    def get_client(cls, id):
        for client in cls.all_clients:
            if client.index == id:
                return client

    def update_d(self, alpha=ALPHA):
        sum_d = 0 
        for neighbor_j in self.neighbor_list:
            sum_d += self.weight*neighbor_j.d
        sum_d += self.weight*self.d
        grad_f = self.gradient()
        self.d = (1 - alpha)*sum_d + alpha*grad_f
        return self.d
    
    def update_x(self, grad_f, T):
        sum_x = 0
        for neighbor_j in self.neighbor_list:
            sum_x += self.weight*neighbor_j.x
        sum_x += self.weight*self.x 

        v = argmax_v(grad_f=grad_f, k=K)

        self.x = sum_x + v/T
        return self.x
    
    def gradient(self, num_sample = NUM_SAMPLE):
        grad_f = gradient_estimate(self.x, num_sample, self.matrix, self.f_dict)
        return grad_f

    def action(self, T):
        d = self.update_d()
        x = self.update_x(grad_f=d, T=T)
        return x, d