import os
import pickle
from collections import OrderedDict

with os.scandir("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/user_inputs_new_ids/") as dir:
    file_names={}
    for entry in dir:
        file_names[int(entry.name[:-5])] = entry.path
    file_names = OrderedDict(sorted(file_names.items()))



# with open("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/partial_graphs.p", "rb") as file:
#         j = pickle.load(file)

# print(j[:5])

