import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import operator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.lookup_table = {}
        self.rewards_table = []
        self.positiveReward = 0
        self.negativeReward = 0
        self.state = None
        self.done = False
        self.action = None
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        
        total = self.positiveReward + self.negativeReward
        self.rewards_table.append((self.positiveReward, self.negativeReward, total, self.done))

        self.planner.route_to(destination)
        self.positiveReward = 0
        self.negativeReward = 0
        self.done = False

    def get_action(self):
        if self.state not in self.lookup_table.keys():
            self.lookup_table[self.state] = {None: 1.0, 'forward': 1.0, 'left': 1.0, 'right': 1.0}
            return random.choice(self.lookup_table[self.state].keys())

        return max(self.lookup_table[self.state].iteritems(), key=operator.itemgetter(1))[0]

    def get_max_q(self, state):
        if state not in self.lookup_table.keys():
            self.lookup_table[state] = {None: 1.0, 'forward': 1.0, 'left': 1.0, 'right': 1.0}

        return max(self.lookup_table[state].values())

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        # TODO: Update state
        self.state = (inputs['oncoming'], inputs['light'], inputs['left'], inputs['right'], self.next_waypoint)

        # TODO: Select action according to your policy
        self.action = self.get_action()

        # Execute action and get reward
        reward = self.env.act(self, self.action)
        self.done = self.env.done
        if reward > 0:
            self.positiveReward += reward
        else:
            self.negativeReward += reward

        # TODO: Learn policy based on state, action, reward
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        next_state = (inputs['oncoming'], inputs['light'], inputs['left'], inputs['right'], self.next_waypoint)
        self.learn_q(self.action, reward, next_state)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}"\
        #     .format(deadline, inputs, a, r)  # [debug]

    def learn_q(self, action, reward, next_state):
        """d: refers to the destination state"""
        max_q = self.get_max_q(next_state)
        self.lookup_table[self.state][action] += self.alpha * (reward - self.gamma * max_q -
                                                                  self.lookup_table[self.state][action])


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    env = Environment()  # create environment (also adds some dummy traffic)
    agent = env.create_agent(LearningAgent)  # create agent
    agent.alpha = 0.9
    agent.gamma = 0.1
    env.set_primary_agent(agent, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(env, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
