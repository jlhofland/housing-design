import json
import torch
import torch.nn as nn

def parse_input_json(file_path):
    # Read in input data from JSONC file, "./input.jsonc"
    with open(file_path, 'r') as openfile:
        # Reading from json file into python dict
        layout = json.load(openfile)
    print("ssasd")
    print(layout)

    room_number_data = torch.zeros(6)
    room_number_data[0] = layout["number_of_living_rooms"]
    room_number_data[1] = int(layout["living_rooms_plus?"])
    room_number_data[2] = layout["number_of_bedrooms"]
    room_number_data[3] = int(layout["bedrooms_plus?"])
    room_number_data[4] = layout["number_of_bathrooms"]
    room_number_data[5] = int(layout["bathrooms_plus?"])

    exterior_walls_sequence = torch.tensor(layout["exterior_walls"], dtype=torch.float32)
    connections_sequence = torch.tensor(layout["connections"], dtype=torch.float32)

    return room_number_data, exterior_walls_sequence, connections_sequence

room_number_data, exterior_walls_sequence, connections_sequence = parse_input_json("./input.jsonc")

# Encode the wall and connection sequences with LSTMs
# num_hidden_units refers to the number of features in the short-term memory and thus the final output vector
lstm_hidden_units = 64  # Adjust as needed

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[-1, :]  # Take the last output of the sequence

# Encode the sequences
exterior_walls_encoder = LSTMEncoder(input_dim=4, hidden_dim=lstm_hidden_units)
connections_encoder = LSTMEncoder(input_dim=4, hidden_dim=lstm_hidden_units)

exterior_walls_encoded = exterior_walls_encoder(exterior_walls_sequence)
connections_encoded = connections_encoder(connections_sequence)

# Concatenate the vectors
final_conditioning_vector = torch.cat((room_number_data, exterior_walls_encoded, connections_encoded), dim=0)[None, :]

print(final_conditioning_vector.shape)

# # Examine encoder structure, weights
# for params in exterior_walls_encoder.state_dict().keys():
#     print(params)
#     print(exterior_walls_encoder.state_dict()[params].shape)

