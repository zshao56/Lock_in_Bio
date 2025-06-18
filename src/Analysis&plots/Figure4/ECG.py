import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import data_path

PPG_path = data_path("Figure4/ECG/re-ECG.csv")
df = pd.read_csv(PPG_path, header=None, names=["counts", "lockin", "regular", "reference"], skiprows=1, sep="\t")

ECG_lockin = df["lockin"]
ECG_regular = df["regular"]
ECG_reference = df["reference"]

plt.figure(figsize=(10, 6))
plt.plot(ECG_lockin, label="lockin")
plt.plot(ECG_regular + 10, label="regular")
plt.plot(ECG_reference + 20, label="reference")
plt.xlabel("Time")
plt.ylabel("ECG")
# plt.yscale('log')

plt.legend()
plt.savefig(data_path("Figure4/ECG/ECG.svg"))
plt.show()
