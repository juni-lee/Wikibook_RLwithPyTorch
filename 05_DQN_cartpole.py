import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import namedtuple
import random
import torch
from torch import nn
from torch import optim


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0


    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)



class Brain:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(self.num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32,32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, self.num_actions))

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])


        self.model.eval()

        # Q(s,a)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # reward + gamma * max(Q(s',a'))
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_action_values = torch.zeros(BATCH_SIZE)
        next_state_action_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + GAMMA * next_state_action_values

        # model train
        self.model.train()
        loss = torch.nn.functional.smooth_l1_loss(state_action_values,
                                                  expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)

        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action



class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)



class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)


    def run(self):
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        episode_final = False

        for episode in range(NUM_EPISODES):
            # Initialize the environment at the beginning of episode
            observation = self.env.reset()

            # get 1st state
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):

                # get action
                action = self.agent.get_action(state, episode)

                # action에 의한 state transition
                observation_next, _, done, _ = self.env.step(action.item())

                # get next_state
                if done:
                    state_next = None
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                else:
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                # get reward
                if done:
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1
                else:
                    reward = torch.FloatTensor([0.0])

                # push (state, action, state_next, reward) to ReplayMemory
                self.agent.memorize(state, action, state_next, reward)

                # NN update
                self.agent.update_q_function()

                if done:
                    print("{} Episode: Finished after {} steps, 최근 10 에피소드의 평균 단계 수 = {}".format(episode,
                                                                                                 step + 1,
                                                                                                 episode_10_list.mean()))
                    break
                else:
                    state = state_next

            if complete_episodes >= 10:
                print('10 에피소드 연속 성공')
                episode_final = True
                break



if __name__ == '__main__':
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    ENV = 'CartPole-v0'
    GAMMA = 0.99
    MAX_STEPS = 200
    NUM_EPISODES = 300

    BATCH_SIZE = 64
    CAPACITY = 5000
    lr = 0.0001

    cartpole_env = Environment()
    cartpole_env.run()

    print()