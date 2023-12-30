from environment import Environment
from agent import Agent
from experience_replay import ExperienceReplay
import time

if __name__ == "__main__":

    grid_size = 5

    environment = Environment(grid_size = grid_size, render_on = True)
    agent = Agent(grid_size = grid_size, epsilon = 1, epsilon_decay = 0.998, epsilon_end = 0.01)

    experience_replay = ExperienceReplay(capacity = 10000, batch_size = 32)

    episodes = 5000

    max_steps = 200

    for episode in range(episodes):

        state = environment.reset()

        for step in range(max_steps):
            print('Episode:' episode)
            print('Step:', step)
            print('Epsilon:', agent.epsilon)

            action = agent.get_action(state)
            reward, next_state, done = environment.step(action)

            experience_replay.add_experience(state, action, reward, next_state, done)

            if experience_replay.can_provide_sample():
                experiences = experience_replay.sample_batch()
                agent.learn(experiences)
            
            state = next_state

            if done:
                break
    agent.save(f'models/model_{grid_size}.h5')