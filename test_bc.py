import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from keras.models import load_model
import gym


model_path = 'bc.h5'
model = load_model(model_path)

print(model.summary())
env = gym.make("Hopper-v2")

returns = []
observations = []
actions = []
max_steps = 1000

for i in range(20):

    print('iter', i)
    obs = env.reset()
    print("shape of obs after env reset is before reshape   : ", obs.shape)
    # obs.reshape(1, obs.shape[0])
    done = False
    totalr = 0.
    steps = 0
    while not done:
        obs = obs[None, :]
        print("obs shape: ", obs.shape)
        action = model.predict(obs)
        print("shape of action is: ", action.shape)
        # print("type of action is: ", type(action))
        # print("action is: ", action)
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        print("shape of obs after step is: ", obs.shape)
        totalr += r
        steps += 1
        env.render()
        if steps % 100 == 0: print("%i/%i" % (steps, max_steps))

        if steps >= max_steps:
            break
    returns.append(totalr)

print('returns', returns)
print('mean return', np.mean(returns))
print('std of return', np.std(returns))

# expert_data = {'observations': np.array(observations),
#                'actions': np.array(actions)}
# print("dumping path is: " + os.path.join('expert_data', args.envname + '.pkl'))
#
# with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
#     pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

