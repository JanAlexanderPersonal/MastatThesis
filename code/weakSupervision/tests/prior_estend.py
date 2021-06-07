import torch

def vectorized_distance(A, p):
    H,W = A.shape
    px, py = p
    X,Y = np.ix_(np.arange(H),np.arange(W))
    return np.sqrt((X-px)**2 + (Y-py)**2)


troch.ones((6,200,200))
