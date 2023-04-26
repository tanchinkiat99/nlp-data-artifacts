import json
import matplotlib.pyplot as plt
import numpy as np

with open("./results/confidence.json") as f:
    data = json.load(f)

confidence = [0 for e in data["1.0"]]
num_epochs = len(data)
for epoch in data:
    for i in range(len(confidence)):
        confidence[i] += (np.exp(data[epoch][i]) / num_epochs)

easy_idx = []
hard_idx = []
for idx, score in enumerate(confidence):
    if score > 45.0:
        easy_idx.append(idx)
    elif score < 40.0:
        hard_idx.append(idx)

classes = {}
classes["easy"] = easy_idx
classes["hard"] = hard_idx

with open("./results/classification.json", "w") as wf:
    json.dump(classes, wf)


# Graph
plt.hist(confidence, bins=120, edgecolor="black", density=False)
plt.title("Confidence")
plt.xlabel("Confidence score")
plt.ylabel("Number of data examples")
plt.savefig("./results/confidenceplot.png")
