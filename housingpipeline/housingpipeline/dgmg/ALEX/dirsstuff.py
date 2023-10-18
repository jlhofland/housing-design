import os
import pickle
from collections import OrderedDict

# with os.scandir("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/user_inputs_new_ids/") as dir:
#     file_names={}
#     for entry in dir:
#         file_names[int(entry.name[:-5])] = entry.path
#     file_names = OrderedDict(sorted(file_names.items()))



# with open("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/partial_graphs.p", "rb") as file:
#     j = pickle.load(file)
#     with open("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/partial_graphs_reduced.p", "wb") as file2:
#         pickle.dump(j[:500], file2)


# with open("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/completed_graphs.p", "rb") as file:
#     j = pickle.load(file)
#     with open("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/completed_graphs_reduced.p", "wb") as file2:
#         pickle.dump(j[:500], file2)



with open("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/partial_graphs_reduced.p", "rb") as file:
    seq = pickle.load(file)
    with open("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/dgmg/ALEX/partial_graphs_reduced.txt", "w") as txt:
        for home in seq[:10]:
            txt.write("\n**************** NEW HOME ****************\n")
            for dec in home:
                txt.write(str(dec)+"\n")

with open("/home/evalexii/Documents/IAAIP/datasets/dgmg_datasets/completed_graphs_reduced.p", "rb") as file:
    seq = pickle.load(file)
    with open("/home/evalexii/Documents/IAAIP/housing-design/housingpipeline/housingpipeline/dgmg/ALEX/completed_graphs_reduced.txt", "w") as txt:
        for home in seq[:10]:
            txt.write("\n**************** NEW HOME ****************\n")
            for dec in home:
                txt.write(str(dec)+"\n")
