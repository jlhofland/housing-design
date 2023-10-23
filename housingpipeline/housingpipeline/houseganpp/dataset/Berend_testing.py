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

# with open("housingpipeline/housingpipeline/houseganpp/dataset/HHGPP_train_data.npy", "rb") as f:
#     b = pickle.load(f)

# ''' Check for angled EW and for type 0 rooms '''
# Houses_with_thick_walls = []

# for counter,home in enumerate(b):
#     all_rooms_type0 = False
#     # Find EW ids
#     EW_lst = []
#     false_wall = False
#     for i,node in enumerate(home[1]):
#         # print(node)
#         if node[0] == 1:
#             EW_lst.append(i)

#     if len(EW_lst) == len(home[1]):
#         all_rooms_type0 = True
#     # Get EW bbs that are not alligned with axis / have width
#     EW_bb_lst = home[0][EW_lst[0]:] * 256
    
#     for bb in EW_bb_lst:
#         x0, y0, x1, y1 = bb
#         width = x1 - x0
#         height = y1 - y0    
#         if width > 1 and height > 1:
#             false_wall = True
#             # print(bb)
#     if false_wall == True:
#         Houses_with_thick_walls.append([counter, all_rooms_type0])
#     print(counter)

# for boi in Houses_with_thick_walls:
#     print(boi)
# print(len(Houses_with_thick_walls))

# abc=0
# for n in Houses_with_thick_walls:
#     if n[1] == True:
#         abc+=1
# print(abc)

with open("housingpipeline/housingpipeline/houseganpp/dataset/HHGPP_eval_data_filtered.p", "rb") as f:
    data = pickle.load(f)

direction_list = []
all_room_distribution = []

for counterz,home in enumerate(data):
    print(counterz)
    num_directions_per_room = []
    for i, room in enumerate(home[1]): # Check each room
        room_direction_list = []
        num_directions = 0
        North = 0
        East = 0
        South = 0
        West = 0

        if room[0] == 0.: # Not EW
            for j,edge in enumerate(home[2]): # Check each edge
                if edge[1] == 1: # If edge exists
                    if edge[0] == i: # If edge is from given room
                        room_direction_list.append(home[3][j][2])
            # print(f'direction going out of room {i}')
            # print(room_direction_list)
            # print('')
            
            for direction in room_direction_list:
                if direction in {7,0,1}: # direction is SE, E, NE
                    East = 1
                if direction in {1,2,3}: # direction is NE, N, NW
                    North = 1
                if direction in {3,4,5}: # direction is NW, W, SW
                    West = 1
                if direction in {5,6,7}: # direction is SW, S, SE
                    South = 1
            num_directions = East + North + West + South

            direction_list.append(room_direction_list)
            num_directions_per_room.append(num_directions)

        # Printing for checking
        # print(home[1])
        # for j,eee in enumerate(home[2]):
        #     print(f'{eee}  + {home[3][j]}')
    
    room_distribution = []
    for i in range(5):
        room_distribution.append(num_directions_per_room.count(i) / len(num_directions_per_room))
    
    all_room_distribution.append(room_distribution)





average_direction_distributions = np.array([0,0,0,0,0])

for distr in all_room_distribution:
    average_direction_distributions = average_direction_distributions + np.array(distr)

average_direction_distributions = average_direction_distributions / len(all_room_distribution)


print(all_room_distribution)
print(average_direction_distributions)

'''

Distribution of amount of rooms with amount of directions checked:
[0, 1, 2, 3, 4]
[0, 3.3926383920000003e-04, 5.467806213999999e-03, 6.611541502e-02, 9.280775147999998e-01]

'''