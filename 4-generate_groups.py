NUM_ROUNDS = 20
NUM_USERS = 55540
LAYER_1_SIZE = 10
LAYER_2_SIZE = 10
GROUP_SIZE = 500

print("Importing Libraries...")
import numpy as np
import math
from tqdm import trange
import networkx as nx
import time

print("Importing Data...")
mtrxAcluster = []
mtrxBcluster = []
for i in trange(NUM_ROUNDS):
    mtrxAcluster.append(np.load('./preds/cluster/layer1/' + str(i) + '.npy'))
    mtrxBcluster.append(np.load('./preds/cluster/layer2/' + str(i) + '.npy'))
mtrxAcluster = np.vstack(mtrxAcluster)
mtrxBcluster = np.vstack(mtrxBcluster)

print("Processing Data...")
mtrxBcluster2 = []
for j in range(NUM_USERS):
    model2 = j % LAYER_2_SIZE
    pos2 = math.floor(j / LAYER_2_SIZE)
    users_per_round2 = NUM_USERS // LAYER_2_SIZE
    mtrxBcluster2.append(mtrxBcluster[:,model2*users_per_round2 + pos2])
mtrxBcluster2 = np.vstack(mtrxBcluster2)

print("Grouping Users...")
start_time = time.time()
def clusterUser(i):
    pred1C = mtrxAcluster[i]
    pred2C = mtrxBcluster2[i]
    pred = pred1C + pred2C
    return np.flip(np.argsort(pred))
G = nx.Graph()
G.add_nodes_from(range(NUM_USERS))
for i in trange(NUM_USERS):
    pred = clusterUser(i)
    if (pred[0] != i):
        for j in pred[0:3]:
             G.add_edge(i, j)
cc = nx.connected_components(G)
cc = [c for c in cc if len(c) > 1]
cc.sort(key=len)
groups = []
group = set([])
for c in cc:
    group = group.union(c)
    if (len(group) >= GROUP_SIZE):
        groups.append(group)
        group = set([])
end_time = time.time()

print('Created ' + str(len(groups)) + ' groups of size', [len(group) for group in groups])
file = open('./data/groups.txt','w')
file.write("\n".join([",".join(map(str, group)) for group in groups]))
file.close()

print("Finished in %s Minutes" % ((end_time - start_time) / 60))
file = open("./stats/grouping.txt", "w")
file.write(str(end_time - start_time))
file.close()
