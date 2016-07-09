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
        self.Q = {}
        self.A = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.positiveReward = 0
        self.negativeReward = 0
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # self.alpha *= 0.9
        # self.gamma /=0.9
        total = self.positiveReward + self.negativeReward
        print "LearningAgent.update(): gamma = {}, alpha = {}, positive={}, negative={}, total={}" \
            .format(self.alpha, self.gamma, self.positiveReward, self.negativeReward, total)

        self.positiveReward = 0
        self.negativeReward = 0

    def get_action(self):
        if self.s not in self.Q.keys():
            self.Q[self.s] = {None: 1.0, 'forward': 1.0, 'left': 1.0, 'right': 1.0}
            return random.choice(self.Q[self.s].keys())

        return max(self.Q[self.s].iteritems(), key=operator.itemgetter(1))[0]

    def get_max_q(self, s):
        if s not in self.Q.keys():
            self.Q[s] = {None: 1.0, 'forward': 1.0, 'left': 1.0, 'right': 1.0}

        return max(self.Q[s].values())

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        # TODO: Update state
        self.s = (inputs['oncoming'], inputs['light'], inputs['left'], inputs['right'], self.next_waypoint)

        # TODO: Select action according to your policy
        a = self.get_action()

        # Execute action and get reward
        r = self.env.act(self, a)
        if r > 0:
            self.positiveReward += r
        else:
            self.negativeReward += r

        # TODO: Learn policy based on state, action, reward
        inputs = self.env.sense(self)
        d = (inputs['oncoming'], inputs['light'], inputs['left'], inputs['right'], self.next_waypoint)
        self.learn_q(a, r, d)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}"\
        #     .format(deadline, inputs, a, r)  # [debug]

    def learn_q(self, a, r, d):
        """d: refers to the destination state"""
        max_q = self.get_max_q(d)
        self.Q[self.s][a] += self.alpha * (r - self.gamma * max_q - self.Q[self.s][a])


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
