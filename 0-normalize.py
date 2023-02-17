print("Importing Libraries...")
import time
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

headers = open("./data/headers.csv").read().split(",")
users = open("./data/users.txt").read().split("\n")

def standardizednp(trainDF: pd.DataFrame, validateDF: pd.DataFrame, clusterDF: pd.DataFrame, testDF: pd.DataFrame) -> tuple:
    trainSize = trainDF.shape[0]
    validateSize = validateDF.shape[0]
    clusterSize = clusterDF.shape[0]

    df = pd.concat([trainDF, validateDF, clusterDF, testDF])
    df['uid'] = pd.factorize(df['uid'])[0]
    df.drop('nid', axis=1, inplace=True)
    cols = df.columns
    cols = cols.delete(list(range(7)))
    ct = ColumnTransformer([
        ('StandardScaler', StandardScaler(), cols)
    ], remainder='passthrough')
    datas = ct.fit_transform(df)

    trainData = torch.tensor(datas[:trainSize])
    validateData = torch.tensor(datas[trainSize:trainSize+validateSize])
    clusterData = torch.tensor(datas[trainSize+validateSize:trainSize+validateSize+clusterSize])
    testData = torch.tensor(datas[trainSize+validateSize+clusterSize:])

    trainData = torch.cat([trainData[:, -7:], trainData[:, :-7]], dim=1)
    validateData = torch.cat([validateData[:, -7:], validateData[:, :-7]], dim=1)
    clusterData = torch.cat([clusterData[:, -7:], clusterData[:, :-7]], dim=1)
    testData = torch.cat([testData[:, -7:], testData[:, :-7]], dim=1)

    return trainData, validateData, clusterData, testData

print("Importing Data...")
trainDF = []
validateDF = []
clusterDF = []
testDF = []
for user in tqdm(users):
    df = pd.read_csv('./data/train/' + str(user) + '.csv', header=None)
    df.columns = headers
    trainDF.append(df)
    df = pd.read_csv('./data/validate/' + str(user) + '.csv', header=None)
    df.columns = headers
    validateDF.append(df)
    df = pd.read_csv('./data/cluster/' + str(user) + '.csv', header=None)
    df.columns = headers
    clusterDF.append(df)
    df = pd.read_csv('./data/test/' + str(user) + '.csv', header=None)
    df.columns = headers
    testDF.append(df)

start_time = time.time()

print("Processing Data...")
trainDF = pd.concat(trainDF)
validateDF = pd.concat(validateDF)
clusterDF = pd.concat(clusterDF)
testDF = pd.concat(testDF)

trainDF.dropna(inplace=True)
validateDF.dropna(inplace=True)
clusterDF.dropna(inplace=True)
testDF.dropna(inplace=True)

print("Normalizing Data...")
trainData, validateData, clusterData, testData = standardizednp(trainDF, validateDF, clusterDF, testDF)

end_time = time.time()
print("Finished in %s Minutes" % ((end_time - start_time) / 60))
file = open("./stats/normalization.txt", "w")
file.write(str(end_time - start_time))
file.close()

print("Saving Results...")
torch.save(trainData, './data/train.pt')
torch.save(validateData, './data/validate.pt')
torch.save(clusterData, './data/cluster.pt')
torch.save(testData, './data/test.pt')
