import numpy as np
import random

class Environment:
    def __init__(
        self, 
        grid_size,
        render_on = False
    ):
        self.grid_size = grid_size
        self.grid = []

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))

        self.agent_location = self.add_agent()
        self.goal_location = self.add_goal()

        if self.render_on:
            self.render()

        return self.get_state()
        
    def add_agent(self):
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

        self.grid[location[0]][location[1]] = 1

        return location

    def add_goal(self):
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size))

        while self.grid[location[0]][location[1]]==1:
            location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size))

        self.grid[location[0]][location[1]] = -1
        
        return location

    def get_state(self):
        state = self.grid.flatten()
        return state

    def render(self):
        grid = self.grid.astype(int).tolist()

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

        old_distance = np.abs(previous_location[0] - self.goal_location[0]) + np.abs(previous_location[1] - self.goal_location[1])

        reward = 0

        move = actions[action]

        new_location = (previous_location[0] + move[0], previous_location[1] + move[1])

        if self.is_valid_location(new_location):
            self.grid[previous_location[0]][previous_location[1]] = 0
            self.grid[new_location[0]][new_location[1]] = 1

            self.agent_location = new_location

            new_distance = np.abs(new_location[0] - self.goal_location[0]) + np.abs(new_location[1] - self.goal_location[1])

            done = False
            
            if self.agent_location == self.goal_location:
                done = True
                reward = 100
            else:
                reward = previous_distance - new_distance - 0.1
        else:
            done = False
            reward = -3

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

    