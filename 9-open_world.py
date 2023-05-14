NUM_ROUNDS = 20
NUM_USERS = 55540
LAYER_1_SIZE = 10
LAYER_2_SIZE = 10

print("Importing Libraries...")
import numpy as np
from tqdm import tqdm, trange
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Importing Data...")
mtrxAtest = []
for i in trange(NUM_ROUNDS):
    mtrxAtest.append(np.load('./preds/test/layer1/' + str(i) + '.npy'))
mtrxAtest = np.vstack(mtrxAtest)

data = []

def predictUser(i):
    pred = mtrxAtest[i][0:49986]
    data.append([
        np.max(pred) / 50,
        np.std(pred) / 0.4,
        np.argmax(pred)
    ])
    return np.argmax(pred)

valid = 0
total = 0
t = tqdm(range(0,5554), desc='0/0 Valid (0%)')
for i in t:
    if (predictUser(i) == i): valid += 1
    total += 1
    t.set_description(str(valid) + "/" + str(total) + " Valid (" + str((valid/total)*100) + "%)")

valid = 0
total = 0
t = tqdm(range(49986,55540), desc='0/0 Valid (0%)')
for i in t:
    if (predictUser(i) == i): valid += 1
    total += 1
    t.set_description(str(valid) + "/" + str(total) + " Valid (" + str((valid/total)*100) + "%)")

y = [True] * 5554 + [False] * 5554
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=0)

clf = LogisticRegression(solver='newton-cg')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print('True Inclusion', sum([pred[i] == True and y_test[i] == True for i in range(len(pred))]) / len(pred))
print('False Inclusion', sum([pred[i] == True and y_test[i] == False for i in range(len(pred))]) / len(pred))
print('True Exclusion', sum([pred[i] == False and y_test[i] == False for i in range(len(pred))]) / len(pred))
print('False Exclusion', sum([pred[i] == False and y_test[i] == True for i in range(len(pred))]) / len(pred))
print('Overall Accuracy', sum([pred[i] == y_test[i] for i in range(len(pred))]) / len(pred))
