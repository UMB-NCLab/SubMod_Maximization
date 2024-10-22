from typing import List, Tuple

import os
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

def download_data(dataset = "mnist", sub_ratio = 1) -> Tuple[Dataset, Dataset]:

    transform = transforms.Compose([transforms.ToTensor(),])

    trainset, testset = None, None
    if dataset == "cifar10":
        trainset = CIFAR10(root="data",train=True,
                download=True,transform = transform)
        testset = CIFAR10(root="data",train=False,
                download=True, transform=transform)
    elif dataset == "mnist":
        trainset = MNIST(root="data",train=True,
                                download=True,transform = transform)
        testset = MNIST(root="data",train=False,
                                download=True, transform=transform)
    elif dataset == "fmnist":
        trainset = FashionMNIST(root="data",train=True,
                                download=True,transform = transform)
        testset = FashionMNIST(root="data",train=False,
                                download=True, transform=transform)
    else:
        raise NotImplementedError

    if sub_ratio < 1:
        raise ValueError('Subset ratio must be >= 1')
    
    train_index = list(range(0, len(trainset), sub_ratio))
    trainset = Subset(trainset, train_index)
    test_index = list(range(0, len(trainset), sub_ratio))
    testset = Subset(testset, test_index)
    return trainset, testset

# pylint: disable=too-many-locals
def partition_data_similarity(
    num_clients, similarity=1.0, seed=42, dataset_name="cifar10", sub_ratio=1
) -> Tuple[List[Dataset], Dataset]:
    """Partition the dataset into subsets for each client.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    similarity: float
        Parameter to sample similar data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    sub_ratio: int, optional
        Used as a step for data subset splitting
    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = download_data(dataset_name, sub_ratio)
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    s_fraction = int(similarity * len(trainset))
    prng = np.random.default_rng(seed)
    idxs = prng.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))

    # sample iid data per client from iid_trainset
    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    if similarity == 1.0:
        return trainsets_per_client, testset

    tmp_t = rem_trainset.dataset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    targets = tmp_t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: List[List] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i % num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    rem_trainsets_per_client: List[List] = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(
                    Subset(rem_trainset.dataset, act_idx)
                )
                ids += 1

    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset(
            [trainsets_per_client[i]] + rem_trainsets_per_client[i]
        )

    return trainsets_per_client, testset

def partition_data_label_quantity(
    num_clients, labels_per_client, seed=42, dataset_name="cifar10", sub_ratio=1
) -> Tuple[List[Dataset], Dataset]:
    """Partition the data according to the number of labels per client.

    Logic from https://github.com/Xtra-Computing/NIID-Bench/.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    num_labels_per_client: int
        Number of labels per client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used
    sub_ratio: int, optional
        Used as a step for data subset splitting
    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """

    trainset, testset = download_data(dataset_name, sub_ratio)
    prng = np.random.default_rng(seed)

    targets = torch.Tensor([trainset.dataset.targets[i] for i in trainset.indices])

    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1 
        if num_clients < num_classes:
            z  = num_classes - num_clients + i
            if z < num_classes:
                current.append(z)
                times[z] += 1
                j += 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients: List[List] = [[] for _ in range(num_clients)] # List of empty lists 
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset

def partition_data_dirichlet(
    num_clients, alpha, seed=42, dataset_name="cifar10", sub_ratio=1
) -> Tuple[List[Dataset], Dataset]:
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used
    sub_ratio: int, optional
        Used as a step for data subset splitting
    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = download_data(dataset_name, sub_ratio)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
    tmp_t = torch.Tensor([trainset.dataset.targets[i] for i in trainset.indices])
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset

def partition_data(
    num_clients, dataset_name, partition ,alpha=0.5, similarity=1.0, labels=1, seed=42, sub_ratio=1):
    
    if partition == "IID":
        partitioned_set, test_set = partition_data_similarity(num_clients=num_clients, similarity=similarity, dataset_name=dataset_name, sub_ratio=sub_ratio)
    elif partition == "label_quantity_1":
        partitioned_set, test_set = partition_data_label_quantity(num_clients=num_clients,labels_per_client = 1, dataset_name=dataset_name, sub_ratio=sub_ratio)
    elif partition == "label_quantity_2":
        partitioned_set, test_set = partition_data_label_quantity(num_clients=num_clients,labels_per_client = 2, dataset_name=dataset_name, sub_ratio=sub_ratio)
    elif partition == "label_quantity_3":
        partitioned_set, test_set = partition_data_label_quantity(num_clients=num_clients,labels_per_client = 3, dataset_name=dataset_name, sub_ratio=sub_ratio)    
    elif partition == "dirichlet":
        partitioned_set, test_set = partition_data_dirichlet(num_clients=num_clients,alpha = alpha, dataset_name=dataset_name, sub_ratio=sub_ratio)    
    else:
        raise NotImplementedError

    label_count = np.zeros(10)
    for i in tqdm(range(num_clients), desc = f"Partitionning {dataset_name}, {partition}"):
        directory = f"./data/{sub_ratio}/{dataset_name}/{num_clients}_clients/{partition}/client_{i}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for j in range(len(partitioned_set[i])):
            label_count[partitioned_set[i][j][1]] += 1
            if dataset_name == "mnist":    
                img = np.squeeze(partitioned_set[i][j][0].numpy())*255
            elif dataset_name == "cifar10": 
                img = partitioned_set[i][j][0].numpy().transpose(1,2,0)*255
            cv.imwrite(os.path.join(directory, f"{partitioned_set[i][j][1]}_{label_count[partitioned_set[i][j][1]]}.jpg"), img)
    
    partitioned_set, test_set = partition_data_similarity(num_clients=num_clients, similarity=similarity, dataset_name=dataset_name, sub_ratio=sub_ratio)
    count = 0
    for i in tqdm(range(num_clients), desc = f"Server collecting {dataset_name}"):
        directory = f"./data/{sub_ratio}/{dataset_name}/server/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for j in range(1, len(partitioned_set[i]), num_clients):
            label_count[partitioned_set[i][j][1]] += 1
            if dataset_name == "mnist":    
                img = np.squeeze(partitioned_set[i][j][0].numpy())*255
            elif dataset_name == "cifar10": 
                img = partitioned_set[i][j][0].numpy().transpose(1,2,0)*255
            cv.imwrite(os.path.join(directory, f"{partitioned_set[i][j][1]}_{label_count[partitioned_set[i][j][1]]}.jpg"), img)
            count += 1
    print(f"count: {count}")
    return partitioned_set, test_set

def download_centralized(dataset_name, sub_ratio):
    trainset, testset = download_data(dataset=dataset_name,sub_ratio=sub_ratio)
    
    label_count = np.zeros(10)

    directory = f"./data/{sub_ratio}/{dataset_name}/centralized/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    count = 0
    for i in tqdm(range(len(trainset)), desc = f"Downloading centralized {dataset_name}"):
        label_count[trainset[i][1]] += 1
        if dataset_name == "mnist":    
            img = np.squeeze(trainset[i][0].numpy())*255
        elif dataset_name == "cifar10": 
            img = trainset[i][0].numpy().transpose(1,2,0)*255
        cv.imwrite(os.path.join(directory, f"{trainset[i][1]}_{label_count[trainset[i][1]]}.jpg"), img)
        count += 1
    print(f"count: {count}")
if __name__ == "__main__":
    num_clis = 5
    k = 20
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "mnist", partition = "IID", sub_ratio=k)
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "mnist", partition = "label_quantity_1", sub_ratio=k)
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "mnist", partition = "label_quantity_2", sub_ratio=k)
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "mnist", partition = "label_quantity_3", sub_ratio=k)
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "mnist", partition = "dirichlet", alpha = 0.5, sub_ratio=k)

    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "cifar10", partition = "IID", sub_ratio=k)
    # # # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "cifar10", partition = "label_quantity_1")
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "cifar10", partition = "label_quantity_2", sub_ratio=k)
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "cifar10", partition = "label_quantity_3", sub_ratio=k)
    # partitioned_set, test_set = partition_data(num_clients = num_clis, dataset_name = "cifar10", partition = "dirichlet", alpha = 0.5, sub_ratio=k)
    download_centralized("cifar10", sub_ratio=5*k)
    # download_centralized("mnist", sub_ratio=k)