import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import data_path

PPG_path = data_path("Figure4/PPG_face.csv")
df = pd.read_csv(PPG_path, header=None, names=["counts", "chin", "cheek", "forehead"], skiprows=1, sep="\t")

PPG_chin = df["chin"]
PPG_cheek = df["cheek"]
PPG_forehead = df["forehead"]

plt.figure(figsize=(10, 6))
plt.plot(PPG_chin, label="Chin")
plt.plot(PPG_cheek, label="Cheek")
plt.plot(PPG_forehead, label="Forehead")
plt.xlabel("Time")
plt.ylabel("PPG")
plt.legend()
plt.savefig(data_path("Figure4/PPG_face.svg"))
plt.show()
