NOTES_TEST = 50
NUM_TREES = 200
NUM_USERS = 55540

print("Importing Libraries...")
import time
import torch
import numpy as np
from lightgbm import LGBMClassifier, log_evaluation
from joblib import load
from tqdm import tqdm

print("Importing Data...")
testData = torch.load('./data/test.pt')
groups = open("./data/groups.txt").read().split("\n")
groups = [list(map(int, group.split(","))) for group in groups]

print("Processing Data...")
def getClassifyData(data):
    dataX = data[:, 1:]
    dataY = data[:, 0]
    return dataX, dataY
testX, testY = getClassifyData(testData)

print("Importing Models...")
clfs3 = []
for i in tqdm(range(len(groups))):
    clfs3.append(load('./models/layer3/model' + str(i) + '.pkl'))

for round in range(len(groups)):
    group = groups[round]
    print("Starting Round " + str(round+1) + "/" + str(len(groups)) + "...")

    print("Testing Accuracy...")
    start_time = time.time()
    mtrxCtest = []
    valid = 0
    total = 0
    t = tqdm(group, desc='0/0 Valid (0%)')
    for i in t:
        preds3 = clfs3[round].predict_proba(testX[50*i:NOTES_TEST+50*i])
        pred3 = preds3.sum(axis=0)
        mtrxCtest.append(pred3)
        if (clfs3[round].classes_[np.argmax(pred3)] == i): valid += 1
        total += 1
        t.set_description(str(valid) + "/" + str(total) + " Valid (" + str((valid/total)*100) + "%)")
    end_time = time.time()
    print("Finished in %s Minutes" % ((end_time - start_time) / 60))
    file = open("./stats/testing/" + str(round) + ".txt", "w")
    file.write(str(end_time - start_time))
    file.close()

    print("Saving Results...")
    np.save('./preds/test/layer3/' + str(round), mtrxCtest)
