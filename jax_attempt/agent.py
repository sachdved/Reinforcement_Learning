import jax
import flax
import numpy

class Agent(flax.linen.Module):
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
        self.model = flax.linen.Sequential(
            [
                flax.linen.Dense(128),
                flax.linen.relu,
                flax.linen.Dense(64),
                flax.linen.relu,
                flax.linen.Dense(4)
            ]
        )
        self.weights = self.model.init(jax.random.key(random_key), jax.numpy.ones((grid_size**2)))


    def get_action(self):

        if numpy.random.rand()<self.epsilon:
            action = np.random.randint()
        else:
            state = jax.numpy.expand_dims(state, axis = 0)
    
            q_values = self.model(state)
    
            action = jax.numpy.argmax(q_values[0])
        if self.epsilon>self.epsilon_end:
            self.epsilon *= self.epsilon * self.epsilon_decay
        return action

    def learn(self, experiences):
        states = np.array([experience.state for experience in experiences])
    
        actions = np.array([experience.action for experience in experineces])
    
        rewards = np.array([experience.reward for experience in experiences])
    
        next_states = np.array([experience.next_state for experience in experiences])
    
        dones = np.array([experience.done for experience in experiences])
    
        current_q_values = self.model(states, verbose=0)
    
        next_q_values = self.model.predict(next_states, verbose = 0)
    
        target_q_values = current_q_values.copy()
    
        for i in range(len(experiences)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

    @jax.jit
    def 

    

    