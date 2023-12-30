import torch
import torch.nn as nn
import numpy as np


class Agent(nn.Module):
    def __init__(
        self,
        grid_size,
        epsilon = 1,
        epsilon_end = 0.01,
        epsilon_decay = 0.998,
        random_key = 0
    ):
        super().__init__()
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model = nn.Sequential(
                nn.Linear(grid_size**2,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,4)
        )
        


    def get_action(self):

        if torch.rand()<self.epsilon:
            action = torch.randint(low=0,high=4)
        else:
            state = state.unflatten(-1)
    
            q_values = self.model(state)
    
            action = torch.argmax(q_values[0])
            
        if self.epsilon>self.epsilon_end:
            self.epsilon *= self.epsilon * self.epsilon_decay
        return action




    

    