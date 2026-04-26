import time
import torch
import numpy as np
import random
from sast.algorithm.unlearning.FedUN import FedUN
from sast.lightsecagg.SecAggMath import SecAggMath


class FedUN_SecAgg_Server(FedUN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.secagg_math = SecAggMath(bit_length=31, scale=1e5)
		self.C_clip = self.params.get('C_clip', 5.0) if self.params else 5.0
		self.online_rate_C = self.params.get('C', 1.0) if self.params else 1.0

	def _generate_mds_matrix(self, U, N):
		W = torch.zeros((U, N), dtype=torch.int64, device=self.device)
		for i in range(U):
			for j in range(N):
				W[i, j] = pow(j + 1, i, self.secagg_math.q)
		return W

	def update_subspace_secagg(self, Z_sum):
		if Z_sum is None:
			return
		U_hat, _ = torch.linalg.qr(Z_sum)
		Q = self.U.T @ U_hat
		A, S, B_t = torch.linalg.svd(Q)
		R = A @ B_t
		U_hat_aligned = U_hat @ R
		U_next_unorth = (1 - self.rho) * self.U + self.rho * U_hat_aligned
		self.U, _ = torch.linalg.qr(U_next_unorth)

	# 【修复1】：加入 is_warmup 标识，防止预热阶段误更新模型
	def train_a_round(self, is_warmup=False):
		com_time_start = time.time()

		unlearn_clients = [c for c in self.client_list if c.unlearn_flag]
		retained_clients = [c for c in self.client_list if not c.unlearn_flag]

		selected_ret = []
		if len(retained_clients) > 0:
			num_ret = max(1, int(len(retained_clients) * self.sampling_rate))
			num_ret = min(len(retained_clients), num_ret)
			selected_ret = random.sample(retained_clients, num_ret)

		protocol_clients = selected_ret + unlearn_clients
		N_lsa = len(protocol_clients)

		surviving_count = max(2, int(N_lsa * self.online_rate_C))
		U_lsa = surviving_count
		T_lsa = max(1, U_lsa // 2)
		if U_lsa <= T_lsa: U_lsa = T_lsa + 1

		W_lsa = self._generate_mds_matrix(U_lsa, N_lsa)

		# 参数解析
		is_secagg_param = self.params.get('use_secagg', True) if hasattr(self, 'params') and self.params else True
		if isinstance(is_secagg_param, str):
			is_secagg_enabled = is_secagg_param.lower() in ['true', '1', 't', 'y', 'yes']
		else:
			is_secagg_enabled = bool(is_secagg_param)

		# 【修复2】：在预热阶段，把发给客户端的模型更新权重置为 0，只收集用于更新 U 的特征矩阵 Z
		beta_to_send = 0.0 if is_warmup else self.beta
		gamma_to_send = 0.0 if is_warmup else self.gamma

		all_Z_shares = {j: {} for j in range(N_lsa)}
		all_delta_shares = {j: {} for j in range(N_lsa)}
		Z_ciphers, delta_ciphers = {}, {}
		Z_meta, delta_meta = None, None

		do_update_U = (self.u_update_freq > 0 and (self.current_comm_round + 1) % self.u_update_freq == 0)

		for idx, client in enumerate(protocol_clients):
			msg = {
				'command': 'cal_secagg_update',
				'U_t': self.U if do_update_U else None,
				'C_clip': self.C_clip,
				'beta': beta_to_send,
				'gamma': gamma_to_send,
				'do_projection': self.do_projection, 'epochs': self.epochs, 'lr': self.lr,
				'target_module': self.module,
				'use_secagg': is_secagg_enabled,
				'U_lsa': U_lsa, 'T_lsa': T_lsa, 'N_lsa': N_lsa, 'W_lsa': W_lsa
			}
			res = client.get_message(msg)
			Z_ciphers[idx] = res.get('Z_i_cipher')
			delta_ciphers[idx] = res.get('delta_i_cipher')

			if is_secagg_enabled:
				if res.get('Z_shares'):
					for j in range(N_lsa): all_Z_shares[j][idx] = res['Z_shares'][j]
					Z_meta = res['Z_meta']
				for j in range(N_lsa): all_delta_shares[j][idx] = res['delta_shares'][j]
				delta_meta = res['delta_meta']

		surviving_indices = random.sample(range(N_lsa), surviving_count)
		decoder_indices = surviving_indices[:U_lsa]
		Z_agg_shares, delta_agg_shares = [], []

		if is_secagg_enabled:
			for j in surviving_indices:
				protocol_clients[j].get_message({
					'command': 'store_shares',
					'Z_shares': all_Z_shares[j],
					'delta_shares': all_delta_shares[j]
				})

			for j in decoder_indices:
				res = protocol_clients[j].get_message(
					{'command': 'aggregate_shares', 'surviving_indices': surviving_indices})
				if res.get('Z_agg') is not None: Z_agg_shares.append(res['Z_agg'])
				if res.get('delta_agg') is not None: delta_agg_shares.append(res['delta_agg'])

		com_time_end = time.time()
		if not hasattr(self, 'communication_time'): self.communication_time = 0.0
		self.communication_time += com_time_end - com_time_start

		cal_time_start = time.time()

		# --- 密文处理分支 ---
		if is_secagg_enabled:
			Z_mask_sum = self.secagg_math.lightsecagg_decode(Z_agg_shares, decoder_indices, W_lsa, U_lsa, T_lsa,
			                                                 Z_meta[0], Z_meta[1]) if Z_agg_shares else None
			delta_mask_sum = self.secagg_math.lightsecagg_decode(delta_agg_shares, decoder_indices, W_lsa, U_lsa, T_lsa,
			                                                     delta_meta[0],
			                                                     delta_meta[1]) if delta_agg_shares else None

			if do_update_U and Z_mask_sum is not None:
				Z_cipher_sum = torch.zeros_like(Z_mask_sum, dtype=torch.int64)
				for i in surviving_indices:
					if Z_ciphers[i] is not None:
						Z_cipher_sum = torch.remainder(Z_cipher_sum + Z_ciphers[i], self.secagg_math.q)
				Z_sum_fq = torch.remainder(Z_cipher_sum - Z_mask_sum, self.secagg_math.q)
				Z_sum_real = self.secagg_math.dequantize_from_finite_field(Z_sum_fq, 1)

				self.update_subspace_secagg(Z_sum_real)

			# 【修复3】：严格保证只有在不是预热阶段时，才更新模型
			if not is_warmup and delta_mask_sum is not None:
				delta_cipher_sum = torch.zeros_like(delta_mask_sum, dtype=torch.int64)
				for i in surviving_indices:
					delta_cipher_sum = torch.remainder(delta_cipher_sum + delta_ciphers[i], self.secagg_math.q)
				delta_sum_fq = torch.remainder(delta_cipher_sum - delta_mask_sum, self.secagg_math.q)
				avg_grad = self.secagg_math.dequantize_from_finite_field(delta_sum_fq, 1)

				self.update_module(self.module, self.optimizer, self.lr, avg_grad)

		# --- 明文处理分支 ---
		else:
			if do_update_U:
				z_list = [Z_ciphers[i] for i in surviving_indices if Z_ciphers[i] is not None]
				if z_list: self.update_subspace_secagg(sum(z_list))

			# 同理，明文模式下预热阶段也不更新模型
			if not is_warmup:
				grad_sum = sum([delta_ciphers[i] for i in surviving_indices])
				self.update_module(self.module, self.optimizer, self.lr, grad_sum)

		cal_time_end = time.time()
		if not hasattr(self, 'computation_time'): self.computation_time = 0.0
		self.computation_time += cal_time_end - cal_time_start

	# 【修复4】：重写 run 函数，让预热阶段走 SECURE 通道
	def run(self):
		for client in self.client_list:
			if client.unlearn_flag:
				client.criterion = self.UnLearningCELoss()

		if self.U is None:
			self.init_subspace()

		# === 受安全聚合保护的预热阶段 ===
		print(f"=== Start SECURE Warm-up Phase ({self.warmup_rounds} rounds, Online Rate C={self.online_rate_C}) ===")
		for j in range(self.warmup_rounds):
			self.current_comm_round = j  # 供 do_update_U 计算频率使用
			self.train_a_round(is_warmup=True)
			print(f"Secure Warm-up Round {j + 1}/{self.warmup_rounds} Done.")
		print("=== Warm-up Phase Finished. Subspace Ready. ===")

		# === 受安全聚合保护的遗忘阶段 ===
		print("=== Start SECURE Unlearning Phase ===")
		self.current_comm_round = 0  # 轮次复位，供遗忘阶段使用

		# 将测试和轮次推进依然交还给父类 while 循环底层的 update_module，防止“跳步”Bug
		while not self.terminated():
			self.train_a_round(is_warmup=False)
