import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np


class PPO:
    """
    NOTICE: the batch_size is much less than the episodes!
    I thought the episodes can be set like 10*batch_size
    """

    def __init__(self, s_dims, a_dims, action_upper_bound,
                 action_lower_bound,
                 actor_lr=0.001,
                 critic_lr=0.002,
                 batch_size=32,
                 GAMA=0.9,
                 epsilon=0.2
                 ):
        # 网络参数的设置
        self.s_dims = s_dims  # 状态的维度
        self.a_dims = a_dims  # 动作的维度
        self.epsilon = epsilon

        # 优化器的定义
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(critic_lr)

        # reward衰减
        self.GAMA = GAMA

        # 动作的上下限
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound

        # 容器的参数
        self.batch_size = batch_size
        self.store_counter = 0

        self.state_batch = np.zeros((self.batch_size, self.s_dims))
        self.action_batch = np.zeros((self.batch_size, self.a_dims))
        self.reward_batch = np.zeros((self.batch_size, 1))

        # actor net
        self.new_actor = self.actor_net()
        self.old_actor = self.actor_net()
        self.old_actor.set_weights(self.new_actor.get_weights())

        # critic net
        self.critic = self.critic_net()

    def actor_net(self):
        # 初始化权值
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.s_dims,))
        # PPO算法速度优于DDPG可以设置大一点的网络
        out = layers.Dense(128, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(128, activation="relu")(out)
        out = layers.BatchNormalization()(out)
        outputs = layers.Dense(self.a_dims, activation="tanh", kernel_initializer=last_init)(out)

        # outputs = outputs * self.action_upper_bound
        model = tf.keras.Model(inputs, outputs)

        return model

    def critic_net(self):
        # 输入为state，输出为v
        # 网络的创建需要优化
        inputs = layers.Input(shape=(self.s_dims,))
        out = layers.Dense(16, activation='relu')(inputs)
        out = layers.Dense(32, activation='relu')(out)
        out_v = layers.Dense(1)(out)

        model = tf.keras.Model(inputs, out_v)
        return model

    def action_policy(self, state):
        sample_actions = tf.squeeze(self.new_actor(state)*self.action_upper_bound)
        sample_actions = sample_actions.numpy()
        legal_action = np.clip(sample_actions, self.action_lower_bound, self.action_upper_bound)
        self.store_counter += 1
        return legal_action

    '''
    store函数返回当前batch是否已满，能不能进行网络的更新
    '''

    def store(self, state, action, reward):
        index = self.store_counter % self.batch_size

        self.state_batch[index] = state
        self.action_batch[index] = action
        self.reward_batch[index] = reward

        if index + 1 == self.batch_size:
            done = True
        else:
            done = False

        return done

    '''
    由critic生成的v
    可能存在一些问题
    '''

    def get_v(self, state):
        v = tf.squeeze(self.critic(state))
        # v = v.numpy()
        return v

    # 这里有几种算法实现，先用简单算法
    # 还有PPO2中的算法
    def discount_reward(self):
        discount_reward = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size - 1):
            discount_reward[i + 1] = self.GAMA * discount_reward[i] + self.reward_batch[i + 1]

        # 归一化
        discount_reward = (discount_reward - np.mean(discount_reward)) / np.std(discount_reward)

        return discount_reward

    def learn(self):
        # 将数据转换为张量
        state_batch = tf.convert_to_tensor(self.state_batch)
        # action_batch = tf.convert_to_tensor(self.action_batch)
        # reward_batch = tf.convert_to_tensor(self.reward_batch)
        # reward_batch = tf.cast(reward_batch, dtype=tf.float32)  # 数据类型转换

        # 求discount_reward
        discount_reward = self.discount_reward()
        discount_reward = tf.convert_to_tensor(discount_reward, dtype=tf.float32)

        # 计算优势函数（PPO1）
        # 存在问题
        # v = self.get_v(self.state_batch)
        # advantage = discount_reward - v
        # adv_tensor = tf.convert_to_tensor(advantage)

        # 神经网络的更新
        # critic的更新
        with tf.GradientTape() as tape:
            v = self.get_v(state_batch)
            y = discount_reward - v
            critic_loss = tf.math.reduce_mean(tf.square(y))
            # print(critic_loss)
            # print("***********************")
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        # new_actor的更新
        with tf.GradientTape() as tape:
            ratio = self.new_actor(state_batch) / (self.old_actor(
                state_batch) + 1e-5)  # 防止0的出现
            adv = discount_reward - self.get_v(state_batch)
            adv_rat = ratio * adv
            # clip = tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon)
            actor_loss = -tf.math.reduce_mean(tf.math.minimum(
                adv_rat,
                tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon)* adv
            ) )
            # print(actor_loss)
            # print("***********************************")

        actor_grad = tape.gradient(actor_loss, self.new_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.new_actor.trainable_variables)
        )

    def update_old_act(self):
        new_weights = []
        target_variables = self.old_actor.weights
        for i, variable in enumerate(self.new_actor.weights):
            new_weights.append(variable * 1 + target_variables[i] * 0)

        self.old_actor.set_weights(new_weights)

    def save_model(self):
        self.new_actor.save_weights("pendulum_online_actor.h5")
        self.old_actor.save_weights("pendulum_online_critic.h5")
        self.critic.save_weights("pendulum_target_actor.h5")

    def clean_data(self):
        self.state_batch = np.zeros((self.batch_size, self.s_dims))
        self.action_batch = np.zeros((self.batch_size, self.a_dims))
        self.reward_batch = np.zeros((self.batch_size, 1))
