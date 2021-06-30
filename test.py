from env2048 import Env2048
from env2048mem import Env2048Mem

env = Env2048Mem()
env2 = Env2048()
print(env.action_spec())
print(env2.action_spec())
print(env.observation_spec())
print(env2.observation_spec())
