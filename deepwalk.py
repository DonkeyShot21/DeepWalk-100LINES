import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

def random_walk(G, v, t):
    rw = [v]
    for j in range(t-1):
        rw.append(random.choice([i for i in G[rw[j]]]))
    return rw

def slide_window(O, rw):
    data = []
    for i in range(len(rw)-w+1):
        s = rw[i:i+w]
        target = s.pop(w//2)
        for context in s:
            data.append([target, context])
    return data

def get_input_layer(word_idx, num_nodes):
    x = torch.zeros(num_nodes).float()
    x[word_idx] = 1.0
    return x

d = 2
epochs = 1000
w = 5
t = 10
G = nx.karate_club_graph()
O = list(G.node)
num_nodes = len(O)

init_lr = 0.1
W1 = Variable(torch.randn(d, num_nodes).float(), requires_grad=True)
W2 = Variable(torch.randn(num_nodes, d).float(), requires_grad=True)

for e in range(epochs):
    running_loss = 0
    lr = init_lr / (1 + 1000*e / epochs)
    random.shuffle(O)
    for v in O:
        rw = random_walk(G, v, t)
        data = slide_window(O, rw)
        for target, context in data:
            x = Variable(get_input_layer(target, num_nodes)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)

            log_softmax = F.log_softmax(z2, dim=0)

            loss = F.nll_loss(log_softmax.view(1,-1), y_true)
            running_loss += loss.item()
            loss.backward()
            W1.data -= lr * W1.grad.data
            W2.data -= lr * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()
    print(e, running_loss/(len(O)*(t-w+1)*(w-1)), sep=":")

embeddings = dict()
for v in O:
    x = Variable(get_input_layer(v, num_nodes)).float()
    emb = np.array(torch.matmul(W1, x).data)
    embeddings[v] = emb

pickle.dump(embeddings, open( "embeddings.pkl", "wb" ) )
