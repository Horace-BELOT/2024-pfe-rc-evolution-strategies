
import os
import sys
from PPO.env import Environnement
from PPO.utils import Action_adapter, Reward_adapter
from PPO.PPO import PPO_agent
import torch

def main():
    # Create the environment
    env: Environnement = Environnement(input_dim=4)

    # Set the parameters
    opt = {
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'net_width': 64,
        'a_lr': 1e-5,
        'c_lr': 1e-5,
        'l2_reg': 1e-4,
        'a_optim_batch_size': 64,
        'c_optim_batch_size': 64,
        'gamma': 0.99,
        'lambd': 0.95,
        'clip_rate': 0.2,
        'K_epochs': 10,
        'T_horizon': 2048,
        'entropy_coef': 0.01,
        'entropy_coef_decay': 0.995,
        'Distribution': 'Beta',
        'dvc': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create the agent
    agent = PPO_agent(**opt)

    total_steps = 0
    for _ in range(opt['K_epochs']):
        idx = 0
        s = env.reset()
        rewards = []
        for i in range(env._max_episode_steps):
            a, logprob_a = agent.select_action(s, deterministic=False)
            # act = Action_adapter(a, 1)  # [0,1] to [-1,1]
            s_next, r, done, _ = env.step(a)
            rewards.append(r)
            
            agent.put_data(s, a, r, s_next, logprob_a, done, False,idx)
            
            s = s_next
            idx += 1
            total_steps += 1
            
            if idx % opt['T_horizon'] == 0:
                agent.train()
                idx = 0
                print(f"Total steps: {total_steps} | Mean rewards: {sum(rewards)/len(rewards)}")
            
            # if total_steps % 1000 == 0:
            #     print(f"Total steps: {total_steps}")
            #     score = evaluate_policy(env, agent, 1, 5)
    env.close()
        
if __name__ == "__main__":
    main()