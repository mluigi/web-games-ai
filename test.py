from tf_agents.environments import utils

from env2048mem import Env2048Mem

env = Env2048Mem()
utils.validate_py_environment(env, episodes=5)
