import numpy as np
from enum import Enum

class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Memory():

    def __init__(self):
        self.action = []
        self.state = []
        self.reward = []
    
    def reset(self):
        self.action = []
        self.state = []
        self.reward = []

    def append(self, action, state, reward):
        self.action.append(action)
        self.state.append(state)
        self.reward.append(reward)

    def get_values(self):
        return self.action, self.state, self.reward

class GridWorld_2():

    def __init__(self, cols, rows, epsilon):
        self.epsilon = epsilon
        self.cols = cols
        self.rows = rows
        self.actions = list(Actions)
        self.goal_state = (self.rows - 1, self.cols - 1)
        self.state = self.start_state()
        self.reward = 0
        self.wall = self.create_random_wall(1)
        self.ice = self.create_random_ice(1)
        self.policy = [0.25, 0.25, 0.25, 0.25]
        self.Q_sa = np.zeros((4,rows,cols))
        for d in range(4):
            for i in range(rows):
                for j in range(cols):
                    self.Q_sa[d,i,j] = np.random.random()
        self.sa = np.zeros((4,rows,cols))
        self.memory = Memory()
    
    def start_state(self):
        x = np.random.randint(0, self.cols)
        y = np.random.randint(0, self.rows)
        while (x, y) == self.goal_state:
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
        return (x,y)

    def reset(self, cols, rows, epsilon):
        self.epsilon = epsilon
        self.cols = cols
        self.rows = rows
        self.actions = list(Actions)
        self.goal_state = (self.rows - 1, self.cols - 1)
        self.state = self.start_state()
        self.reward = 0
        self.wall = self.create_random_wall(1)
        self.ice = self.create_random_ice(1)
        self.policy = [0.25, 0.25, 0.25, 0.25]
        self.Q_sa = np.zeros((4,rows,cols))
        for d in range(4):
            for i in range(rows):
                for j in range(cols):
                    self.Q_sa[d,i,j] = np.random.random()
        self.sa = np.zeros((4,rows,cols))
        self.memory.reset()

    def run(self):
        return self.evaluate()

    def create_random_wall(self, length):
        wall = []
        while len(wall) < length:
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
            if (x, y) == self.goal_state:
                continue
            wall.append((x, y))
        return wall

    def create_random_ice(self, length):
        ice = []
        for i in range(length):
            x = np.random.randint(0, self.cols)
            y = np.random.randint(0, self.rows)
            ice.append((x, y))
        return ice

    def pick_action(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(list(Actions), p=self.policy)
        else:
            x,y = self.state
            action = np.argmax(self.Q_sa[:,y,x])
            return list(Actions)[action]

    def up(self):
        if self.state[0] == 0:
            return self.state
        else:
            return self.state[0] - 1, self.state[1]

    def down(self):
        if self.state[0] == self.rows - 1:
            return self.state
        else:
            return self.state[0] + 1, self.state[1]

    def left(self):
        if self.state[1] == 0:
            return self.state
        else:
            return self.state[0], self.state[1] - 1

    def right(self):
        if self.state[1] == self.cols - 1:
            return self.state
        else:
            return self.state[0], self.state[1] + 1

    def is_final_state(self):
        if self.state == self.goal_state:
            return True
        else:
            return False

    def move(self):
        action = self.pick_action()
        next_state = self.get_next_state(action)
        reward = self.reward
        if next_state in self.wall:
            self.reward -= 10
            self.memory.append(action,self.state,self.reward - reward)
        elif next_state in self.ice:
            self.reward -= 5
            self.memory.append(action,self.state,self.reward - reward)
            self.state = next_state
        elif next_state == self.goal_state:
            self.reward += 100
            self.memory.append(action,self.state,self.reward - reward)
            self.state = next_state
        else:
            self.reward -= 1
            self.memory.append(action,self.state,self.reward - reward)
            self.state = next_state
    def get_next_state(self, action):
        if action == Actions.UP:
            return self.up()
        elif action == Actions.DOWN:
            return self.down()
        elif action == Actions.LEFT:
            return self.left()
        elif action == Actions.RIGHT:
            return self.right()
    
    def reset_memory(self):
        self.memory.reset()

    def evaluate(self):
        while not self.is_final_state():
            #print(f'Current state: {self.state}')
            #print(f'Current policy: {self.policy}')
            #print(f'Current reward: {self.reward}')
            self.move()
            #self.update_policy()

        #print(self.reward)
        return self.reward

def MC_policy_iteration_without_ES(cols = 5,rows = 5, episodes = 1000, epsilon = 0.2, gamma = 0.98):
    Gridworld = GridWorld_2(cols, rows, epsilon)
    for t in range(episodes):
        Gridworld.run()
        G = 0
        for i in range(len(Gridworld.memory.action)):
            G = G * gamma +        