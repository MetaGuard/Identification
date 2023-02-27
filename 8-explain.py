LAYER_1_SIZE = 10
LAYER_2_SIZE = 10

print("Importing Libraries...")
from tqdm import tqdm
from lightgbm import plot_importance
from joblib import load
import matplotlib.pyplot as plt
import numpy as np

print("Importing Models (1/2)...")
clfs1 = []
for i in tqdm(range(LAYER_1_SIZE)):
    clfs1.append(load('./models/layer1/model' + str(i) + '.pkl'))

print("Importing Models (2/2)...")
clfs2 = []
for i in tqdm(range(LAYER_2_SIZE)):
    clfs2.append(load('./models/layer2/model' + str(i) + '.pkl'))

importance = []
print("Analyzing Importance (1/2)...")
for i in tqdm(range(LAYER_1_SIZE)):
    importance.append(clfs1[i]._Booster.feature_importance(importance_type='gain'))
print("Analyzing Importance (2/2)...")
for i in tqdm(range(LAYER_2_SIZE)):
    importance.append(clfs2[i]._Booster.feature_importance(importance_type='gain'))
importance = np.sum(importance, axis=0)

print("Generating Graph...")
feature_names = open("./data/headers.csv").read().split(",")[2:]
colors = ['green']*22 + ['blue']*210
for i in range(22, len(feature_names)):
    sname = feature_names[i]
    if (sname[4:] == "min" or sname[4:] == "max"):
        if (sname[2:3] == "p"):
            colors[i] = 'orange'
total_importance = sum(importance)
relative_importance = [100*absolute/total_importance for absolute in importance]
feature_importance = list(zip(relative_importance, colors, feature_names))
sorted_importance = sorted(feature_importance, key=lambda x: x[0])
sorted_importance.reverse()
values = [f[0] for f in sorted_importance if f[0] > 0]
colors = [f[1] for f in sorted_importance if f[0] > 0]
plt.xlabel('Feature Importance Rank (#)')
plt.ylabel('% of Entropy Gain Explained')
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='green', lw=4)]
plt.legend(custom_lines, ['Static Features', 'Motion Features', 'Context Features'], title="Feature Type", title_fontproperties={'weight': 'bold'})
plt.bar(range(len(values)), values, width=1.0, color=colors)
plt.tight_layout()
plt.savefig('./stats/features.pdf')
plt.savefig('./stats/features.png')
