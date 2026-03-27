import matplotlib.pyplot as plt

labels = ["Clean", "Noise", "Blur", "Low Light"]

baseline = [0.9839, 0.8947, 0.9840, 0.9833]
wdgrl   = [0.9804, 0.8991, 0.9797, 0.9818]

x = range(len(labels))

plt.figure(figsize=(8,5))

plt.plot(x, baseline, marker='o', label="Baseline")
plt.plot(x, wdgrl, marker='s', label="WDGRL")

plt.xticks(x, labels)
plt.ylabel("ROC-AUC")
plt.title("Baseline vs WDGRL Robustness Comparison")
plt.legend()
plt.grid()

plt.savefig("comparison.png", dpi=300)
plt.show()