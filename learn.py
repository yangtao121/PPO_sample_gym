import gym
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras_PPO import PPO

problem = "Pendulum-v0"
env = gym.make(problem)
env.seed(1)

s_dims = env.observation_space.shape[0]
a_dims = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

total_episodes = 100

RL = PPO(s_dims, a_dims, upper_bound, lower_bound, batch_size=20)

ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0
    step = 0
    episode_state = []
    episode_action = []

    while True:
        # tf_pre_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = RL.action_policy(prev_state)
        # print(action)
        state, reward, done, info = env.step(action)
        episode_state.append(state)
        episode_action.append(action)
        # print(action)

        learn_flag = RL.store(prev_state, action, reward, state, 1)
        episodic_reward += reward

        if learn_flag:
            RL.learn()
            RL.update_old_act()

        if done:
            break

        prev_state = state

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
