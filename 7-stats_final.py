NUM_ROUNDS = 20
NUM_USERS = 55540
LAYER_1_SIZE = 10
LAYER_2_SIZE = 10
NUM_THREADS = 32

print("Importing Libraries...")
import numpy as np
import math
from tqdm import tqdm, trange
import networkx as nx
import time
import pandas as pd
from joblib import load

print("Importing Data (1/3)...")
mtrxAtest = []
for i in trange(NUM_ROUNDS):
    mtrxAtest.append(np.load('./preds/test/layer1/' + str(i) + '.npy'))
mtrxAtest = np.vstack(mtrxAtest)

print("Importing Data (2/3)...")
mtrxBtest = []
for i in trange(NUM_ROUNDS):
    mtrxBtest.append(np.load('./preds/test/layer2/' + str(i) + '.npy'))
mtrxBtest = np.vstack(mtrxBtest)

print("Importing Data (3/3)...")
groups = open("./data/groups.txt").read().split("\n")
groups = [list(map(int, group.split(","))) for group in groups]
mtrxCtest = []
clfs3 = []
for i in trange(len(groups)):
    clfs3.append(load('./models/layer3/model' + str(i) + '.pkl'))
    mtrxCtest.append(np.load('./preds/test/layer3/' + str(i) + '.npy'))

print("Processing Data...")
mtrxBtest2 = []
for j in range(NUM_USERS):
    model2 = j % LAYER_2_SIZE
    pos2 = math.floor(j / LAYER_2_SIZE)
    users_per_round2 = NUM_USERS // LAYER_2_SIZE
    mtrxBtest2.append(mtrxBtest[:,model2*users_per_round2 + pos2])
mtrxBtest2 = np.vstack(mtrxBtest2)

print("Testing Accuracy...")
valid = 0
validl1 = 0
validl2 = 0
validl3 = 0
total = 0
totall3 = 0

def predictUser(i):
    global totall3
    global validl1
    global validl2
    global validl3

    pred1 = mtrxAtest[i]
    if np.argmax(pred1) == i: validl1 += 1

    pred2 = mtrxBtest2[i]
    if np.argmax(pred2) == i: validl2 += 1

    pred = pred1 + pred2
    for g in range(len(groups)):
        if i in groups[g]:
            totall3 += 1
            idx = groups[g].index(i)
            pred = np.argmax(mtrxCtest[g][idx])
            id = clfs3[g].classes_[pred]
            if (id == i): validl3 += 1
            return id
    return np.argmax(pred)

t = tqdm(range(NUM_USERS), desc='0/0 Valid (0%)')
for i in t:
    if (predictUser(i) == i): valid += 1
    total += 1
    t.set_description(str(valid) + "/" + str(total) + " Valid (" + str((valid/total)*100) + "%)")

print('Importing Statistics...')
time_grouping = float(open("./stats/grouping.txt").read())
time_normalization = float(open("./stats/normalization.txt").read())
time_featurize = []
for i in range(NUM_THREADS):
    time_featurize.append(float(open("./stats/featurization/" + str(i) + ".txt").read()) / 1000)
time_train_l1 = []
for i in range(LAYER_1_SIZE):
    time_train_l1.append(float(open("./stats/training/layer1/" + str(i) + ".txt").read()))
time_train_l2 = []
for i in range(LAYER_2_SIZE):
    time_train_l2.append(float(open("./stats/training/layer2/" + str(i) + ".txt").read()))
time_train_l3 = []
for i in range(len(groups)):
    time_train_l3.append(float(open("./stats/training/layer3/" + str(i) + ".txt").read()))
time_test = []
for i in range(NUM_ROUNDS):
    time_test.append(float(open("./stats/clustering/" + str(i) + ".txt").read()))
time_test_l3 = []
for i in range(len(groups)):
    time_test_l3.append(float(open("./stats/testing/" + str(i) + ".txt").read()))

print("Final Results:")
def strtime(t):
    (days, r) = divmod(t, 86400)
    (hours, r) = divmod(r, 3600)
    (minutes, seconds) = divmod(r, 60)
    time = ""
    if (days > 0): time = str(int(days)) + "d "
    return time + str(int(hours)) + "h " + str(int(minutes)) + "m " + str(int(seconds)) + "s"

def pct(p):
    return str(round(p*100,2)) + "%"

tab = []
tab.append(['Featurization', NUM_THREADS, strtime(np.average(time_featurize)), strtime(np.max(time_featurize)), 'N/A'])
tab.append(['Normalization', 1, strtime(time_normalization), strtime(time_normalization), 'N/A'])
tab.append(['Layer 1', LAYER_1_SIZE, strtime(np.average(time_train_l1)), strtime(np.sum(time_train_l1)), pct(validl1/total)])
tab.append(['Layer 2', LAYER_2_SIZE, strtime(np.average(time_train_l2)), strtime(np.sum(time_train_l2)), pct(validl2/total)])
tab.append(['Clustering', NUM_ROUNDS, strtime(np.average(time_test)/2), strtime(time_grouping + np.sum(time_test)/2), 'N/A'])
tab.append(['Layer 3', len(groups), strtime(np.average(time_train_l3)), strtime(np.sum(time_train_l3)), pct(validl3/totall3)])
tab.append(['Testing', NUM_ROUNDS, strtime(np.average(time_test_l3) + np.average(time_test)/2), strtime(np.sum(time_test_l3) + np.sum(time_test)/2), pct(valid/total)])

df = pd.DataFrame(tab, columns = ['Stage', 'Models', 'Time (Per Model)', 'Time (Overall)', 'Accuracy'])
print(df)
