import torch
import torch.nn as nn
import numpy as np


class Environment:
    def __init__(
        self, 
        grid_size,
        render_on = False
    ):
        self.grid_size = grid_size
        self.grid = []
        self.render_on = render_on

    def reset(self):
        self.grid = torch.zeros((self.grid_size, self.grid_size))

        self.agent_location = self.add_agent()
        self.goal_location = self.add_goal()

        if self.render_on:
            self.render()

        return self.get_state()
        
    def add_agent(self):
        location = (torch.randint(0, self.grid_size - 1, (1,)), torch.randint(0, self.grid_size - 1, (1,)))
        self.grid[location] = 1

        return location

    def add_goal(self):
        location = (torch.randint(0, self.grid_size - 1, (1,)), torch.randint(0, self.grid_size-1, (1,)))

        while self.grid[location]==1:
            location = (torch.randint(0, self.grid_size - 1, (1,)), torch.randint(0, self.grid_size-1, (1,)))

        self.grid[location] = -1
        
        return location

    def get_state(self):
        state = self.grid.flatten()
        return state

    def render(self):
        grid = self.grid.detach().numpy().astype(int).tolist()

        for row in grid:
            print(row)
        print(' ')

    def move_agent(self, action):
        actions = {
            0: (-1,0),
            1: (0,-1),
            2: (1,0),
            3: (0,1)
        }

        previous_location = self.agent_location

        previous_distance = torch.abs(previous_location[0] - self.goal_location[0]) + torch.abs(previous_location[1] - self.goal_location[1])

        reward = 0

        move = actions[int(action)]

        new_location = (previous_location[0] + move[0], previous_location[1] + move[1])

        if self.is_valid_location(new_location):
            self.grid[previous_location] = 0
            self.grid[new_location] = 1

            self.agent_location = new_location

            new_distance = torch.abs(new_location[0] - self.goal_location[0]) + torch.abs(new_location[1] - self.goal_location[1])

            done = torch.tensor(False)
            
            if self.agent_location == self.goal_location:
                done = torch.tensor(True)
                reward = torch.tensor(100)
            else:
                reward = (previous_distance - new_distance - 0.1)[0]
        else:
            done = torch.tensor(False)
            reward = torch.tensor(-3)

        return reward, done
        
    def is_valid_location(self, location):
        if (location[0]>=0) and (location[0] < self.grid_size) and (location[1]>=0) and (location[1] < self.grid_size):
            return True
        else:
            return False
 

    def step(self, action):
        reward, done = self.move_agent(action)
        next_state = self.get_state()

        if self.render_on:
            self.render()
        return next_state, reward, done

    