import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras_PPO_V2 import PPO
tf.compat.v1.disable_eager_execution()

problem = "Pendulum-v0"
env = gym.make(problem)
env.seed(1)

s_dims = env.observation_space.shape[0]
a_dims = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

# print(s_dims)
total_episodes = 100

RL = PPO(a_dims, s_dims, upper_bound,batch_size=20)

ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    prev_state = env.reset()
    # print(prev_state)
    episodic_reward = 0
    step = 0
    episode_state = []
    episode_action = []

    while True:
        prev_state = np.array(prev_state).reshape((1, s_dims))
        # print(prev_state)
        action = RL.action_policy(prev_state)
        # print(action)
        state, reward, done, info = env.step(action)
        # reward = -reward
        # print(reward)
        episode_state.append(state)
        episode_action.append(action)
        # print(action)

        learn_flag = RL.store(prev_state, action, reward, state)
        episodic_reward += reward
        step += 1

        if learn_flag:
            RL.learn()

        if done:
            break

        prev_state = state

    # print(step)

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

    # 画出回合图
    episode_state = np.array(episode_state)
    episode_action = np.array(episode_action)
    plt.subplot(221)
    plt.plot(episode_state[:, 0])
    plt.xlabel('theta')
    plt.subplot(222)
    plt.plot(episode_state[:, 1])
    plt.xlabel('theta_dot')
    plt.subplot(223)
    plt.plot(episode_state[:, 2])
    plt.xlabel("action")
    plt.subplot(224)
    plt.plot(episode_action)

    fig_name = "./fig/" + str(ep) + ".jpg"
    plt.savefig(fig_name)
    plt.close()

RL.save_model()

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
