import gymnasium as gym
from groverMazeLearner_highway import GroverMazeLearner
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# test
if __name__ == "__main__":
    # choose env
    envtest = gym.make("highway-bs-v0")
    # init learner
    Elliot = GroverMazeLearner(envtest)
    # good hyperparms (hand-tuned)
    hyperp = {'k': 0.1,  # earlier value 0.1
              'alpha': 0.12, # earlier value 0.01, 0.1
              'gamma': 0.99,
              'eps': 0.01,   #not used anywhere?
              'max_epochs': 1000,
              'max_steps': 15,
              'graphics': False}
    # set hyperparms
    Elliot.set_hyperparams(hyperp)

    # TRAIN
    trajectories = Elliot.train()
    
    # Save the model
    # filename = 'model.qasm'
    # Elliot.save_model(filename)
    # Show trajectories
    # for key in trajectories.keys():
    #     print(key, trajectories[key])

    # # final state values
    # print(Elliot.state_vals.reshape((4, 4)))

    # # grover flags
    # for state, flag in enumerate(Elliot.grover_steps_flag):
    #     print(state, '\t', flag)

    # # state-action circuits
    # for s, circ in enumerate(Elliot.acts_circs):
    #     print('action circuit for state ', s)
    #     print(circ.draw())

