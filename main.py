import argparse
import os

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tqdm import tqdm

from env2048 import Env2048


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        prev_action: any = None
        repeated_same_action = 0
        while not time_step.is_last() and repeated_same_action < 10:
            action_step = policy.action(time_step)
            if prev_action == action_step:
                repeated_same_action += 1
            else:
                repeated_same_action = 0
            prev_action = action_step
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def main(evaluate):
    tf.compat.v1.enable_v2_behavior()
    # Mostly copied from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
    # Hyperparameters
    num_iterations = 20000

    collect_steps_per_iteration = 100
    replay_buffer_max_length = 100000

    batch_size = 64
    learning_rate = 1e-3
    log_interval = 200

    num_eval_episodes = 10
    eval_interval = 1000

    # Environment
    # env = Env2048()
    train_py_env = Env2048(evaluate)
    eval_py_env = Env2048(evaluate)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Agent
    fc_layer_params = (100, 50)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step)
    agent.initialize()

    # Replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # Data Collection
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration)

    collect_driver.run()

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # Save tools
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )

    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()
    # Training
    if evaluate:
        print(f"Average return: {compute_avg_return(eval_env, agent.policy, num_eval_episodes)}")
        train_env.station.shutdown()
        eval_env.station.shutdown()
    else:
        agent.train = common.function(agent.train)
        # agent.train_step_counter.assign(0)
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        returns = [avg_return]
        for _ in tqdm(range(global_step.numpy(), num_iterations)):
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_driver.run()

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = tf.compat.v1.train.get_global_step().numpy()

            if step % log_interval == 0:
                train_checkpointer.save(step)
                tqdm.write(f"step = {step}: loss = {train_loss}")

            if step % eval_interval == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                tqdm.write('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true', help="Evaluate the trained network")
    args = parser.parse_args()
    main(args.eval)
