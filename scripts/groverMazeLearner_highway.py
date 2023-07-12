__author__ = 'QRL_team'

from qiskit import *
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
import csv

from qiskit import IBMQ

provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_qasm_simulator')


class GroverMazeLearner:
    """
    Inits a quantum QLearner object for given environment.
    Environment must be discrete and of "maze type", with the last state as the goal
    """
    def __init__(self, env):
        self.env = env  # gym.make("FrozenLake-v0", is_slippery=False)
        # state and action spaces dims
        # self.obs_dim = self.env.observation_space.shape[1]
        self.obs_dim = 2187
        self.acts_dim = self.env.action_space.n
        # dim of qubits register needed to encode all actions
        self.acts_reg_dim = ceil(np.log2(self.acts_dim))
        # optimal number of steps in original Grover's algorithm
        self.max_grover_steps = int(round(
            np.pi / (4 * np.arcsin(1. / np.sqrt(2 ** self.acts_reg_dim))) - 0.5))
        # quality values
        self.state_vals = np.zeros(self.obs_dim)
        # grover steps taken
        self.grover_steps = np.zeros((self.obs_dim, self.acts_dim), dtype=int)
        # boolean flags to signal maximum amplitude amplification reached
        self.grover_steps_flag = np.zeros((self.obs_dim, self.acts_dim), dtype=bool)
        # learner hyperparms (eps still not used)
        self.hyperparams = {'k': -1, 'alpha': 0.05, 'gamma': 0.99, 'eps': 0.01, 'max_epochs': 1000, 'max_steps': 100
                            , 'graphics': True}
        # current state
        self.state, info = self.env.reset()

        #print("self.state inside init: ",self.state)

        # current action
        self.action = 0
        # list of grover oracles
        self.grover_ops = self._init_grover_ops()
        # list of state-action circuits
        self.acts_circs = self._init_acts_circs()
        # qiskit simulator
        self.SIM = Aer.get_backend('qasm_simulator')

    def set_hyperparams(self, hyperdict):
        """
        Set learner's hyperparameters
        :param hyperdict: a dict with same keys as self's
        :return:
        """
        self.hyperparams = hyperdict
        
    def convert(self, list):
        # multiply each integer element with its
        # corresponding power and perform summation
        # res = sum(d * 3**i for i, d in enumerate(list))
        res = list[0]*3**5 + list[1]*3**4 + list[2]*3**3 + list[3]*3**2 + list[4]*3**1 + (list[5]-1)*3**0
        # if list.len() == 7:
        #     return(int(res)%(10**6))
        return int(res)

    def _init_acts_circs(self):
        """
        Inits state-action circuits
        :return: list of qiskit circuits, initialized in full superposition
        """
        circs = [QuantumCircuit(self.acts_reg_dim, name='|as_{}>'.format(i)) for i in range(self.obs_dim)]
        for c in circs:
            c.h(list(range(self.acts_reg_dim)))
        return circs

    def _update_statevals(self, reward, new_state):
        """
        Bellman equation for state values update
        :param reward: instantaneous reward received by the agent
        :param new_state: state reached upon taking previous action
        :return:
        """
        #print("self.state inside update intervals: ",self.state)
        #print("gamma: ",self.hyperparams['gamma'])
        #print("self.state_vals: ",self.state_vals[new_state])
        self.state_vals[self.convert(self.state[0])] += self.hyperparams['alpha']*(reward
                                                                  + self.hyperparams['gamma']*self.state_vals[self.convert(new_state[0])]
                                                                  - self.state_vals[self.convert(self.state[0])])

    def _eval_grover_steps(self, reward, new_state):
        """
        Choose how many grover step to take based on instantaneous reward and value of new state
        :param reward: the instantaneous reward received by the agent
        :param new_state: the new state visited by the agent
        :return: number of grover steps to be taken,
        if it exceeds the theoretical optimal number the latter is returned instead
        """
        steps_num = int(self.hyperparams['k']*(reward + self.state_vals[self.convert(new_state[0])]))
        return min(steps_num, self.max_grover_steps)

    def _init_grover_ops(self):
        """
        Inits grover oracles for the actions set
        :return: a list of qiskit instructions ready to be appended to circuit
        """
        states_binars = [format(i, '0{}b'.format(self.acts_reg_dim)) for i in range(self.acts_dim)]
        targ_states = [Statevector.from_label(s) for s in states_binars]
        grops = [GroverOperator(oracle=ts) for ts in targ_states]
        return [g.to_instruction() for g in grops]

    def _run_grover(self):
        """
        DEPRECATED
        :return:
        """
        # deploy grover ops on acts_circs
        gsteps = self.grover_steps[self.state, self.action]
        circ = self.acts_circs[self.state]
        op = self.grover_ops[self.action]
        for _ in range(gsteps):
            circ.append(op, list(range(self.acts_reg_dim)))
        self.acts_circs[self.state] = circ

    def _run_grover_bool(self):
        """
        Update state-action circuits based on evaluated steps
        :return:
        """
        flag = self.grover_steps_flag[self.convert(self.state[0]), :]
        gsteps = self.grover_steps[self.convert(self.state[0]), self.action]
        circ = self.acts_circs[self.convert(self.state[0])]
        op = self.grover_ops[self.action]
        if not flag.any():
            for _ in range(gsteps):
                circ.append(op, list(range(self.acts_reg_dim)))
        if gsteps >= self.max_grover_steps and not flag.any():
            self.grover_steps_flag[self.convert(self.state[0]), self.action] = True
        self.acts_circs[self.convert(self.state[0])] = circ

    def _take_action(self):
        """
        Measures the state-action circuit corresponding to current state and decides next action
        :return: action to be taken, int
        """
        #print('state: {}'.format(self.state))
        #print('state: ',self.state)
        # print("self.state: ",self.state)
        # print("self.state convert: ", self.convert(self.state[0]))
        circ = self.acts_circs[self.convert(self.state[0])]
        circ_tomeasure = circ.copy()
        circ_tomeasure.measure_all()
        # circ_tomeasure = transpile(circ_tomeasure)
        # print(circ.draw())
        job = execute(circ_tomeasure, backend=self.SIM, shots=1)
        result = job.result()
        counts = result.get_counts()
        # print("counts: ",counts)
        action = int((list(counts.keys()))[0], 2)
        # print("action: ",action)
        return action

    def save_model(self, filename):
        np.savez(filename,
                 state_vals=self.state_vals,
                 grover_steps=self.grover_steps,
                 grover_steps_flag=self.grover_steps_flag)

    def load_model(self, filename):
        data = np.load(filename)
        self.state_vals = data['state_vals']
        self.grover_steps = data['grover_steps']
        self.grover_steps_flag = data['grover_steps_flag']

    
    def train(self):
        """
        groverize and measure action qstate -> take corresp action
        obtain: newstate, reward, terminationflag
        update stateval, grover_steps
        for epoch in epochs until max_epochs is reached
        :return:
        dictionary of trajectories
        """
        traj_dict = {}
        flag = 0
        # print("observation space:", self.env.observation_space.shape[1])
        arr = []
        total_rewards = []
        steps_arr = []
        
        filename = 'csv_train_10_20.csv'
        # set initial max_steps
        optimal_steps = self.hyperparams['max_steps']
        print("self.actions: ", self.env.action_space.n)
        for epoch in range(self.hyperparams['max_epochs']):
            info_copy = {}
            # if epoch % 10 == 0:
            print("Processing epoch {} ...".format(epoch))
            # reset env
            steps = 0
            self.state, info = self.env.reset()
            # print("self.state at the start of each episode: ",self.state)
            # init list for traj
            traj = [self.state]
            rewards_arr = []
            done = False
            if self.hyperparams['graphics']:
                self.env.render()
            while not done:
                #print('Taking step {0}/{1}'.format(step, optimal_steps), end='\r')
                # print('STATE: ', self.state)
                # Select action
                self.action = self._take_action()
                # take action
                if(self.action == 15):
                    continue
                new_state, reward, terminated, truncated, info = self.env.step(self.action)
                # print("new_state: ", new_state)
                done = terminated or truncated
                # print("done: ", done)
                # print("reward: ", reward)
                # print("info: ", info)
                
                # info_copy = info
                info_copy['speed'] = info['speed']
                info_copy['action'] = info['action']
                info_copy['other_vehicle_collision'] = info['other_vehicle_collision']
                info_copy['agents_ho_prob'] = info['agents_ho_prob'][0]
                info_copy['agents_tran_all_rewards'] = info['agents_tran_all_rewards'][0]
                info_copy['agents_tele_all_rewards'] = info['agents_tele_all_rewards'][0]
                info_copy['agents_rewards'] = info['agents_rewards'][0]
                info_copy['agents_collided'] = info['agents_collided'][0]
                info_copy['distance_travelled'] = info['distance_travelled'][0]
                info_copy['agents_survived'] = info['agents_survived']
                info_copy['episode'] = epoch
                info_copy['total_reward'] = info['agents_rewards'][0]
                info_copy['tran_reward'] = info['agents_tran_all_rewards'][0]
                info_copy['tele_reward'] = info['agents_tele_all_rewards'][0]
                # if new_state == self.state:
                #     # print("reached final state. steps taken: ", steps)
                #     reward -= 10
                #     done = True
                # if new_state == self.obs_dim - 1:
                #     reward += 100 # earlier value 10
                # elif not done:
                #     reward -= 1
                        
                #print(' REWARD: ', reward)
                steps = steps + 1
                rewards_arr.append(reward)
                # update statevals and grover steps
                self._update_statevals(reward, new_state)
                self.grover_steps[self.convert(self.state[0]), self.action] = self._eval_grover_steps(reward, new_state)
                # amplify amplitudes with grover
                # self._run_grover()
                self._run_grover_bool()
                # render if curious
                if self.hyperparams['graphics']:
                    self.env.render()
                # save transition
                traj.append(new_state)
                # if done:
                #     break
                self.state = new_state
                values = list(info_copy.values())
                keys = list(info_copy.keys())
                
                
                if flag == 0:
                    with open(filename, 'w', newline='') as file:
                        flag = 1
                        writer = csv.writer(file)
                        writer.writerow(keys)
                # Append the values as a row in the CSV file
                with open(filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(values)
            
            #print("average rewards: ", np.average(rewards_arr))
            steps_arr.append(steps)
            # print("setps array: ",steps_arr)
            arr.append(np.average(rewards_arr))
            total_rewards.append(np.sum(rewards_arr))
            #print("total_rewards: ",np.sum(rewards_arr))
            traj_dict['epoch_{}'.format(epoch)] = traj
        
        # numbers_series = pd.Series(steps_arr)
        # print("average steps: ", np.average(steps_arr))
        # rolling_mean_steps = numbers_series.rolling(100).mean()
        # fig1, ax1 = plt.subplots()
        # ax1.plot(numbers_series, color = 'violet')
        # ax1.set_xlabel("Episodes")
        # ax1.set_ylabel("Steps")
        # ax1.set_title("learning rate: 0.2")
        # plt.show()
        
        # numbers_series = pd.Series(arr)
        # print("average steps: ", np.average(steps_arr))
        # print("arr:", arr)
        # # print("flag_arr:", flag_arr)
        # rolling_mean_rewards = numbers_series.rolling(100).mean()
        # plt.plot(rolling_mean_rewards, color = 'violet')
        # plt.show()
        
        # last = arr[0]  # First value in the plot (first timestep)
        # smoothed = []
        # for point in arr:
        #     smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        #     smoothed.append(smoothed_val)                        # Save it
        #     last = smoothed_val

        # fig1, ax1 = plt.subplots()
        # ax1.plot(smoothed, color = 'violet')
        # ax1.set_xlabel("Episodes")
        # ax1.set_ylabel("Average Rewards")
        # # ax1.set_title("learning rate: 0.12")
        # plt.show()
        
        # fig1, ax1 = plt.subplots()
        # ax1.plot(arr, color = 'violet')
        # ax1.set_xlabel("Episodes")
        # ax1.set_ylabel("Average Rewards")
        # ax1.set_title("learning rate: 0.01")
        # plt.show()
        
        
        # plt.plot(rolling_mean_steps)
        # plt.show()
        
        #arr.append(np.average(rewards_arr))
        #print("arr: ",len(arr))
        # print("traj_dict: ", traj_dict)
        # print("\n")
        # numbers_series = pd.Series(total_rewards)
        # rolling_mean = numbers_series.rolling(100).mean()

        # last = steps_arr[1]  # First value in the plot (first timestep)
        # smoothed = []
        # for point in steps_arr:
        #     smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        #     smoothed.append(smoothed_val)                        # Save it
        #     last = smoothed_val

        # print("rolling mean: ",rolling_mean)
        # plt.plot(smoothed)
        # plt.show()
        # return trajectories
        return traj_dict
