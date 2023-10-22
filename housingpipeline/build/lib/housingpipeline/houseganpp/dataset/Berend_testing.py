import numpy as np
import pickle


# home = [[6.0, 2.0, 4.0], [np.array([132,   6, 148,  65]), np.array([110,  68, 208, 130]), np.array([132,  91, 160, 130])], [[108, 89, 130, 89, 8, 0], [108, 66, 130, 66, 0, 8], [108, 66, 108, 89, 8, 0], [209, 66, 209, 131, 0, 8], [130, 4, 149, 4, 0, 6], [130, 89, 161, 89, 8, 4], [130, 4, 130, 66, 6, 0], [149, 4, 149, 66, 0, 6], [130, 89, 130, 131, 4, 0], [161, 89, 161, 131, 2, 4], [130, 66, 149, 66, 6, 8], [149, 66, 209, 66, 0, 8], [130, 131, 161, 131, 4, 0], [161, 131, 209, 131, 8, 0]], [[1], [1], [1], [1], [0], [1, 2], [0], [0], [2], [1, 2], [0, 1], [1], [2], [1]], [6, 7, 9, 2, 3], 'floorplan_high_res/0d/00/464243c9d2ffd5507b24bdced40a/0001.jpg']

# def is_edge_adjacent(edgeA, boxB, threshold = 0.03):
#     xa0, ya0, xa1, ya1, _, _ = edgeA
#     xb0, yb0, xb1, yb1 = boxB

#     wa, wb = xa1 - xa0, xb1 - xb0
#     ha, hb = ya1 - ya0, yb1 - yb0

#     xca, xcb = (xa0 + xa1) / 2.0, (xb0 + xb1) / 2.0
#     yca, ycb = (ya0 + ya1) / 2.0, (yb0 + yb1) / 2.0
    
#     delta_x = np.abs(xcb - xca) - (wa + wb) / 2.0
#     delta_y = np.abs(ycb - yca) - (ha + hb) / 2.0

#     if ha == 0:
#         return delta_x < threshold 

#     if wa == 0:
#         return delta_y < threshold
    
# print(is_edge_adjacent([108, 66, 130, 66, 8, 0], np.array([110,  68, 208, 130])))

# with open("houseganpp/dataset/HHGPP_train_data.npy", "rb") as f:
#     b = pickle.load(f)
# print(b)

