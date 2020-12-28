import matplotlib.pyplot as plt
import os

with open("stage_1_2.txt","r") as f:
    loss = f.readlines()

for i in range(len(loss)):
    loss[i] = float(loss[i])
print (loss)
plt.plot(loss)
plt.show()