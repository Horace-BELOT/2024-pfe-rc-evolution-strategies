from PPO.utils import BetaActor, GaussianActor_musigma, GaussianActor_mu, Critic
import numpy as np
import copy
import torch
import math
from pyESN import Torch_ESN_without_input

class PPO_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		if self.Distribution == 'Beta':
			self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_ms':
			self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_m':
			self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		else: print('Dist Error')
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Trajectory holder
		self.s_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.r_hoder = np.zeros((self.T_horizon, 1),dtype=np.float32)
		self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		self.dw_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy.Making the performance more stable
				a = self.actor.deterministic_act(state)
				return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				a = dist.sample()
				a = torch.clamp(a, 0, 1)
				logprob_a = dist.log_prob(a).cpu().numpy().flatten()
				return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)


	def train(self):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		s = torch.from_numpy(self.s_hoder).to(self.dvc)
		a = torch.from_numpy(self.a_hoder).to(self.dvc)
		r = torch.from_numpy(self.r_hoder).to(self.dvc)
		s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)
		dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (~dw) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
				distribution = self.actor.get_dist(s[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

	def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
		self.s_hoder[idx] = s
		self.a_hoder[idx] = a
		self.r_hoder[idx] = r
		self.s_next_hoder[idx] = s_next
		self.logprob_a_hoder[idx] = logprob_a
		self.done_hoder[idx] = done
		self.dw_hoder[idx] = dw

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
		self.critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))
  


class PPO_ESN_agent(object):
	def __init__(self, opt_PPO, opt_ESN, opt_General):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(opt_PPO)
		self.__dict__.update(opt_ESN)
		self.__dict__.update(opt_General)

		self.Torch_ESN_without_input = Torch_ESN_without_input(**opt_ESN)

		# Choose distribution for the actor
		if self.Distribution == 'Beta':
			self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_ms':
			self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_m':
			self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		else: print('Dist Error')
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# Build Trajectory holder
		self.s_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.r_hoder = np.zeros((self.T_horizon, 1),dtype=np.float32)
		self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		self.dw_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
		
		self.hstate = torch.zeros((self.action_dim,), dtype=torch.float32).to(self.dvc)
		self.label_hoder = torch.zeros((self.T_horizon, self.n_outputs),dtype=torch.float32, requires_grad=True).to(self.dvc)
		self.out_hoder = torch.zeros((self.T_horizon, self.n_outputs),dtype=torch.float32, requires_grad=True).to(self.dvc)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				# only used when evaluate the policy.Making the performance more stable
				a = self.actor.deterministic_act(state)
				return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state)
				a = dist.sample()
				a = torch.clamp(a, 0, 1)
				logprob_a = dist.log_prob(a).cpu().numpy().flatten()
				return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)

	def act(self, action):
        # transform it to (1, action_dim) 
		action_tensor = torch.tensor(action).unsqueeze(0).to(self.dvc)
		out, self.hstate = self.Torch_ESN_without_input(action_tensor,state=self.hstate)
		out = out.flatten()
		return out

	def train(self):
		self.entropy_coef*=self.entropy_coef_decay

		'''Prepare PyTorch data from Numpy data'''
		s = torch.from_numpy(self.s_hoder).to(self.dvc)
		a = torch.from_numpy(self.a_hoder).to(self.dvc)
		r = torch.from_numpy(self.r_hoder).to(self.dvc)
		s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).to(self.dvc)
		dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

		''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
		with torch.no_grad():
			vs = self.critic(s)
			vs_ = self.critic(s_next)

			'''dw for TD_target and Adv'''
			deltas = r + self.gamma * vs_ * (~dw) - vs
			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			'''done for GAE'''
			for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
		esn_optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))
		# for i in range(self.K_epochs):

		# 	#Shuffle the trajectory, Good for training
		# 	perm = np.arange(s.shape[0])
		# 	np.random.shuffle(perm)
		# 	perm = torch.LongTensor(perm).to(self.dvc)
		# 	s, a, td_target, adv, logprob_a = \
		# 		s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

		# 	'''update the actor'''
		# 	for i in range(a_optim_iter_num):
		# 		index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
		# 		distribution = self.actor.get_dist(s[index])
		# 		dist_entropy = distribution.entropy().sum(1, keepdim=True)
		# 		logprob_a_now = distribution.log_prob(a[index])
		# 		ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

		# 		surr1 = ratio * adv[index]
		# 		surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
		# 		a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

		# 		self.actor_optimizer.zero_grad()
		# 		a_loss.mean().backward()
		# 		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
		# 		self.actor_optimizer.step()

		# 	'''update the critic'''
		# 	for i in range(c_optim_iter_num):
		# 		index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
		# 		c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
		# 		for name,param in self.critic.named_parameters():
		# 			if 'weight' in name:
		# 				c_loss += param.pow(2).sum() * self.l2_reg

		# 		self.critic_optimizer.zero_grad()
		# 		c_loss.backward()
		# 		self.critic_optimizer.step()

		'''update the ESN'''
		for i in range(esn_optim_iter_num):
			index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))
			self.Torch_ESN_without_input.optimizer.zero_grad()
			esn_loss = self.Torch_ESN_without_input.criterion(self.out_hoder[index], self.label_hoder[index])
			esn_loss.backward(retain_graph=True)
			self.Torch_ESN_without_input.optimizer.step()
			

	def put_data(self, s, a, r, s_next, logprob_a, done, dw, label, out, idx):
		self.s_hoder[idx] = s
		self.a_hoder[idx] = a
		self.r_hoder[idx] = r
		self.s_next_hoder[idx] = s_next
		self.logprob_a_hoder[idx] = logprob_a
		self.done_hoder[idx] = done
		self.dw_hoder[idx] = dw
		self.label_hoder[idx] = label
		self.out_hoder[idx] = out

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
		self.critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))