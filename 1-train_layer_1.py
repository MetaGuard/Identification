NOTES_TRAIN = 150
NOTES_VALIDATE = 3
NUM_TREES = 200
NUM_ROUNDS = 10
NUM_USERS = 55540

print("Importing Libraries...")
import time
import torch
import numpy as np
from lightgbm import LGBMClassifier, log_evaluation
from joblib import dump

print("Importing Data...")
trainData = torch.load('./data/train.pt')
validateData = torch.load('./data/validate.pt')

def getClassifyData(data):
    dataX = data[:, 1:]
    dataY = data[:, 0]
    return dataX, dataY

for round in range(NUM_ROUNDS):
    print("Starting Round " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")

    print("Selecting Data...")
    trainFrame = []
    validateFrame = []
    users_per_round = NUM_USERS // NUM_ROUNDS
    for id in range(users_per_round*round, users_per_round*(round+1)):
        trainFrame.append(trainData[150*id:NOTES_TRAIN+150*id])
        validateFrame.append(validateData[5*id:NOTES_VALIDATE+5*id])

    print("Processing Data...")
    trainX, trainY = getClassifyData(torch.cat(trainFrame))
    validateX, validateY = getClassifyData(torch.cat(validateFrame))

    print("Training Model " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")
    clf = LGBMClassifier(boosting_type='goss', colsample_bytree=0.6933333333333332, learning_rate=0.1, \
        max_bin=63, max_depth=-1, min_child_weight=7, min_data_in_leaf=20, \
        min_split_gain=0.9473684210526315, n_estimators=NUM_TREES, \
        num_leaves=33, reg_alpha=0.7894736842105263, reg_lambda=0.894736842105263, \
        subsample=1, n_jobs=16, objective='multiclass', device_type='gpu')
    start_time = time.time()
    clf.fit(trainX, trainY.long(),
           eval_set=[(validateX, validateY)],
           eval_metric='multi_error',
           callbacks=[log_evaluation()])
    end_time = time.time()
    print("Training Finished in %s Minutes" % ((end_time - start_time) / 60))

    print("Saving Model " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")
    dump(clf, './models/layer1/model' + str(round) + '.pkl')
    file = open("./stats/training/layer1/" + str(round) + ".txt", "w")
    file.write(str(end_time - start_time))
    file.close()
