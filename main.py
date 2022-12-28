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
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10, discount=0.99):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        step = 0
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            cost = np.sum(np.abs(action))
            avg_reward += reward
            avg_cost += discount ** step * cost
            step += 1

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes, D4RL score: {d4rl_score:.3f}, Constraint Value: {avg_cost:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--algorithm", default="BCQ_L")  # Policy name
    parser.add_argument("--env", default="hopper-medium-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--constraint_threshold", default=500.0)
    # BCQ-L
    parser.add_argument("--phi", default=0.05)
    # CPQ
    parser.add_argument("--alpha", default=1.0)
    args = parser.parse_args()

    file_name = f"{args.algorithm}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.algorithm}, Env: {args.env}, Seed: {args.seed}")
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
        policy = BCQ_L.BCQ_L(state_dim, action_dim, max_action, threshold=args.constraint_threshold, phi=args.phi)
        algo_name = f"{args.algorithm}_phi-{args.phi}"
    elif args.algorithm == 'CPQ':
        policy = CPQ.CPQ(state_dim, action_dim, max_action, threshold=args.constraint_threshold, phi=args.phi)
        algo_name = f"{args.algorithm}_alpha-{args.alpha}"

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    evaluations = []
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            evaluations.append(eval_policy(policy, args.env, args.seed, mean, std, discount=args.discount))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
