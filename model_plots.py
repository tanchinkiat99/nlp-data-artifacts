import json
import matplotlib.pyplot as plt


with open("./results/models.json", "r") as rf:
    data = json.load(rf)

accuracy_3_epochs = data["accuracy_3_epochs"]
accuracy_6_epochs = data["accuracy_6_epochs"]
orig_accuracy_3_epochs = data["orig_accuracy_3_epochs"]
orig_accuracy_6_epochs = data["orig_accuracy_6_epochs"]
hard_ex_proportions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

plt.scatter(hard_ex_proportions, accuracy_3_epochs, label="With modified datasets (3 epochs)")
plt.plot(hard_ex_proportions, accuracy_3_epochs)
plt.scatter(hard_ex_proportions, accuracy_6_epochs, label="With modified datasets (6 epochs)")
plt.plot(hard_ex_proportions, accuracy_6_epochs)
plt.scatter(0.351, orig_accuracy_3_epochs, label="With original dataset (3 epochs)")
plt.scatter(0.351, orig_accuracy_6_epochs, label="With original dataset (6 epochs)")
plt.xlabel("Proportion of hard to learn examples")
plt.ylabel("Model Accuracy")
plt.legend(loc="upper right")
plt.ylim(0.45, 0.90)
plt.savefig("./results/accuracyplot.png")