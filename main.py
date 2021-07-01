from __future__ import absolute_import, division, print_function

import argparse
import logging
import logging.config
import os

import tensorflow as tf
import tf_agents
from matplotlib import pyplot as plt
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, ParallelPyEnvironment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tqdm import tqdm

from env2048 import Env2048


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    prev_best_score = 0
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
            if environment.envs[0].best_score > prev_best_score:
                prev_best_score = environment.envs[0].best_score
        total_return += episode_return

    avg_return = total_return / num_episodes

    return avg_return.numpy()[0], prev_best_score


def main(argv):
    tf.compat.v1.enable_v2_behavior()
    logging.config.dictConfig({
        'version': 1,
        # Other configs ...
        'disable_existing_loggers': True
    })
    argv = argv[0]

    evaluate = argv.eval

    # Mostly copied from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
    # Hyperparameters
    num_iterations = argv.num_iterations

    collect_steps_per_iteration = argv.collect_steps_per_iteration
    replay_buffer_max_length = 100000

    batch_size = argv.batch_size
    learning_rate = 2.5e-5
    log_interval = argv.log_interval

    num_atoms = 256
    min_q_value = 0
    max_q_value = 256
    n_step_update = argv.n_step_update
    gamma = 0.99

    num_eval_episodes = 10
    eval_interval = argv.eval_interval

    save_interval = argv.save_interval
    n_parallels = argv.n_parallels

    if evaluate:
        n_parallels = 1

    # Environment
    train_py_env = ParallelPyEnvironment(
        [lambda: Env2048(evaluate)] * n_parallels,
        start_serially=False
    )
    eval_py_env = Env2048(evaluate)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Agent
    fc_layer_params = (64, 64, 32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # q_net = q_network.QNetwork(
    #     train_env.observation_spec(),
    #     train_env.action_spec(),
    #     fc_layer_params=fc_layer_params)
    # agent = dqn_agent.DqnAgent(
    #     train_env.time_step_spec(),
    #     train_env.action_spec(),
    #     q_network=q_net,
    #     optimizer=optimizer,
    #     td_errors_loss_fn=common.element_wise_squared_loss,
    #     train_step_counter=global_step)

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params
    )
    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        train_step_counter=global_step
    )
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

    # Checkpointer
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
                tqdm.write(f"step = {step}: loss = {train_loss}")

            if step % save_interval == 0:
                train_checkpointer.save(step)

            if step % eval_interval == 0:
                avg_return, best_eval_score = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                tqdm.write(
                    f'step = {step}: Average Return = {avg_return}, best score reached in training = '
                    f'{max(list(map(lambda env: env.best_score, train_env.envs)))}'
                    f', best score in eval = {best_eval_score}'
                )
                returns.append(avg_return)
        steps = range(0, num_iterations + 1, eval_interval)
        plt.plot(steps, returns)
        plt.ylabel('Average Return')
        plt.xlabel('Step')

    train_env.close()
    eval_env.close()
    train_py_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true', help="Evaluate the trained network")
    parser.add_argument("--num_iterations", type=int, default=20000)
    parser.add_argument("--collect_steps_per_iteration", type=int, default=20)
    parser.add_argument("--n_step_update", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--n_parallels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    tf_agents.system.multiprocessing.handle_main(main, [args])
