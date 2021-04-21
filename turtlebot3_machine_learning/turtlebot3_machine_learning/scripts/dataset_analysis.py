import numpy as np

a = np.load('dataset_cali.npy',allow_pickle=True)

action_distribution = np.zeros(20)

for data in a:
    action = data[1]
    index = int(action[0]/0.1)+1
    action_distribution[index]+=1

print(action_distribution)