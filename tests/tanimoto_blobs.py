import numpy as np
import random

def generate_data(num_clusters, pops, d, n, flip_prob=0.1, seed=None):
    """
    Generate data for the CLASSIX_T dataset

        Parameters: \n
        num_clusters (int)  : number of clusters \n
        pops (list)         : the popcount of the different cluster centers. len(pops) must be equal to num_clusters \n
        d (int)             : dimension of the data \n
        n (int)             : number of total samples per cluster \n
        seed (int)          : seed for the random number generator \n
    """
    # assert len(pops)>1, "You must provide at least 2 cluster mean populations in order to generate meaningful data"
    if len(pops) != num_clusters:
        try:
            print("The number of mean populations is not equal to the number of clusters. Trying to generate mean populations keeping in mind the minimum and maximum pop counts provided by the user")
            pops = np.linspace(min(pops), max(pops), num_clusters)
        except:
            raise ValueError("The number of cluster centers must be equal to the number of cluster populations")
    
    size_flag = [pops[x]<d for x in range(len(pops))]
    
    assert all(size_flag), "The number of dimensions must be greater than the pop counts of the clusters"
    
    clustc = []
    data = []
    if seed != None:
        np.random.seed(seed)
        random.seed(seed)

    labels = []

    for i in range(num_clusters):
        pop = pops[i]
        p1 = pop/d
        clustc.append(np.random.choice([0, 1], size=d, p=[1-p1, p1]))
        # data.append(clustc[-1])
        labels.append(i)
        for j in range(n):
            labels.append(i)
            flip_indices = np.random.choice([0, 1], size = d, p = [1-flip_prob, flip_prob])
            new_sample = np.logical_xor(clustc[-1], flip_indices).astype(np.int32)
            data.append(new_sample)
            
    return np.array(data).astype(np.int32), np.array(labels).astype(np.int32), clustc