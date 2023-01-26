import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import BCQ_L, CPQ


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, constraint_threshold, seed_offset=100, eval_episodes=10, discount=0.99):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    tot_reward = 0.
    tot_cost = 0.
    discounted_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        step = 0
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            cost = np.sum(np.abs(action))
            tot_reward += reward
            tot_cost += cost
            discounted_cost += discount ** step * cost
            step += 1

    tot_reward /= eval_episodes
    tot_cost /= eval_episodes
    discounted_cost /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(tot_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes, D4RL score: {d4rl_score:.3f}, Total return: {tot_reward:.3f}, "
          f"Constraint Value (discounted): {discounted_cost:.3f}, Constraint Value (undiscounted): {tot_cost:.3f}, Constraint Threshold: {constraint_threshold:.3f}")
    print("---------------------------------------")
    return tot_reward, discounted_cost, tot_cost


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--algorithm", default="CPQ")  # Policy name
    parser.add_argument("--env", default="hopper-medium-replay-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--constraint_threshold", default=683, type=float)
    # BCQ-L
    parser.add_argument("--phi", default=0.05)
    # CPQ
    parser.add_argument("--alpha", default=10)
    args = parser.parse_args()

    save_dir = f"./results/{args.algorithm}_{args.env}_{args.discount}_{args.constraint_threshold}_{args.seed}.txt"
    print("---------------------------------------")
    print(f"Policy: {args.algorithm}, Env: {args.env}, Seed: {args.seed}, Gamma: {args.discount}, Cost_limit: {args.constraint_threshold}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.algorithm == 'BCQ_L':
        policy = BCQ_L.BCQ_L(state_dim, action_dim, max_action, discount=args.discount, threshold=args.constraint_threshold, phi=args.phi)
        algo_name = f"{args.algorithm}_phi-{args.phi}"
    elif args.algorithm == 'CPQ':
        policy = CPQ.CPQ(state_dim, action_dim, max_action, discount=args.discount, threshold=args.constraint_threshold, alpha=args.alpha)
        algo_name = f"{args.algorithm}_alpha-{args.alpha}"

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    eval_log = open(save_dir, 'w')
    # Start training
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            average_return, discounted_cost, _ = eval_policy(policy, args.env, args.seed, mean, std, args.constraint_threshold, discount=args.discount)
            eval_log.write(f'{t + 1},{average_return},{discounted_cost}\n')
            eval_log.flush()
    eval_log.close()