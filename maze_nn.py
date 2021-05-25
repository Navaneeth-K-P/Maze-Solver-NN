import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
import sys
import numpy as np
import math
import random
import time as tm
import gym
import gym_maze
import datetime
env = gym.make("maze-random-10x10-v0")




MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
INPUT_SIZE=int(np.prod(MAZE_SIZE,dtype=int))
NUM_OBSERVATIONS=int(np.prod(MAZE_SIZE))
NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2
DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
NUM_ACTIONS = env.action_space.n
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, NUM_OBSERVATIONS))) #input layer
model.add(Dense(20, activation='relu')) #hidden layer
model.add(Dense(env.action_space.n, activation='linear')) #output layer
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def state_to_input(ob):
    s=np.zeros((1,INPUT_SIZE),dtype=int)
    location=(MAZE_SIZE[0]*int(ob[0]))+int(ob[1])
    s=np.asmatrix(s)
    s[0,location]=1
    #print(s.shape)
    #s.reshape(1,INPUT_SIZE)
    return(s)

def select_action(s,explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(model.predict(s)))
        #print(s.shape)
    return action





def train():

    NUM_EPISODES=500000
    MAX_T=1000
    SOLVED_T=100
    STREAK_TO_END=100
    num_streaks=0
    for episode in range(NUM_EPISODES):
        obv=env.reset()
        s=state_to_input(obv);
        d=False
        total_reward=0
        learning_rate=get_learning_rate(0)
        explore_rate=get_explore_rate(0)
        discount_factor=0.99
        for t in range(MAX_T):
            a=select_action(s,explore_rate)
            obv,r1,d,_=env.step(a)
            s1=state_to_input(obv)
            total_reward+=r1
            #print(s1.shape)
            #print("lol")
            #print(s1)
            best_q=np.amax(model.predict(s1))
            target=model.predict(s)
            target_new=r1+discount_factor*best_q
            target[0,a]=target_new
            target=np.asarray(target)
            model.fit(s,target,verbose=0)

            s=s1
            env.render()

            if env.is_game_over():
                sys.exit()

            if(d):
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break
            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

            if num_streaks > STREAK_TO_END:
                break

            explore_rate = get_explore_rate(episode)
            learning_rate = get_learning_rate(episode)

        if num_streaks > STREAK_TO_END:
            break

def simulate():
    obv=env.reset()
    s=state_to_input(obv)
    d=False
    reward=0
    time=0
    env.render()
    tm.sleep(1)
    while(not d):
        action = int(np.argmax(model.predict(s)))
        obv,r1,d,_=env.step(action)
        env.render()
        tm.sleep(1)
        s1=state_to_input(obv)
        s=s1
        reward+=r1
        time+=1
    print("Simulation ended at time %d with total reward = %f."
          % (time, reward))

train()
while(True):
    d=input("Simulate?(y/n)")
    if(d=='y'):
        simulate()
    else:
        break

#print(obv)
#print(state_to_input([2.0, 1.0]))
#tf.reset_default_graph()
