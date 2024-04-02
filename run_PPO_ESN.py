
import os
import sys
from PPO.env import Environnement
from PPO.utils import Action_adapter, Reward_adapter
from PPO.PPO import PPO_ESN_agent
import torch

def main(input_size,reservoir_size=50):
    # Create the environment
    env: Environnement = Environnement(input_dim=input_size, reservoir_size=reservoir_size)

    # Set the parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt_PPO = {
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'net_width': 128,
        'a_lr': 2e-5,
        'c_lr': 2e-5,
        'l2_reg': 1e-5,
        'a_optim_batch_size': 64,
        'c_optim_batch_size': 64,
        'gamma': 0.99,
        'lambd': 0.95,
        'clip_rate': 0.2,
        'entropy_coef': 0.01,
        'entropy_coef_decay': 0.995,
        'Distribution': 'Beta',
        'dvc': device
    }
    
    opt_ESN = {
        'spectral_radius':0.8,
        'n_outputs':10,
        'n_reservoir':reservoir_size,
        'sparsity':0.5,
        'silent':False,
        'input_scaling':0.7,
        'feedback_scaling':0.2,
        'wash_out':25,
        'learning_rate':0.0003,
        'batch_size':16,
        'device':device
    }
    opt_General = {
        'K_epochs': 10,
        'T_horizon': 2048,
    }

    # Create the agent
    agent = PPO_ESN_agent(opt_PPO, opt_ESN, opt_General)

    total_steps = 0
    for _ in range(10):
        idx = 0
        s = env.reset()
        rewards = []
        for i in range(env._max_episode_steps):
            a, logprob_a = agent.select_action(s,deterministic=False)
            out = agent.act(a)
            # act = Action_adapter(a, 1)  # [0,1] to [-1,1]
            s_next, r, done, _ = env.step(out.cpu().detach().numpy())
            label = torch.tensor(env.y_train[env.current_step])
            rewards.append(r)

            agent.put_data(s, a, r, s_next, logprob_a, done, False,label, out, idx)
            
            s = s_next
            idx += 1
            total_steps += 1
            
            if idx % opt_General['T_horizon'] == 0:
                agent.train()
                idx = 0
                print(f"Total steps: {total_steps} | Mean rewards: {sum(rewards)/len(rewards)}")
                rewards = []
            
            # if total_steps % 1000 == 0:
            #     print(f"Total steps: {total_steps}")
            #     score = evaluate_policy(env, agent, 1, 5)
    env.close()
        
if __name__ == "__main__":
    main(4)