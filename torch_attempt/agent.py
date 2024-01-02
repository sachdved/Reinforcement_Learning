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
        random_key = 0,
        gamma = 0.99
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
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,4),
            #nn.Softmax(dim=-1),
        )
        self.gamma = gamma
        


    def get_action(self, state):

        if torch.rand((1,))<self.epsilon:
            action = torch.randint(low=0,high=4, size=(1,))[0]
        else:
            state = state.unsqueeze(0)
            
            q_values = self.model(state)
    
            action = torch.argmax(q_values.squeeze())
            
        if self.epsilon>self.epsilon_end:
            self.epsilon = self.epsilon * self.epsilon_decay
        return action




    

    