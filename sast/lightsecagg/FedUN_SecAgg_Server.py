import time
import torch
import numpy as np
import random
import os
from sast.algorithm.unlearning.FedUN import FedUN
from sast.lightsecagg.SecAggMath import SecAggMath


class FedUN_SecAgg_Server(FedUN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.secagg_math = SecAggMath(bit_length=31, scale=1e5)
		self.online_rate_C = self.params.get('C', 1.0) if self.params else 1.0
		self.UR = self.max_comm_round
		if hasattr(self, 'params') and self.params is not None:
			self.params['UR'] = self.max_comm_round

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

	def train_a_round(self, is_warmup=False):
		com_time_start = time.time()

		unlearn_clients = [c for c in self.client_list if c.unlearn_flag]
		retained_clients = [c for c in self.client_list if not c.unlearn_flag]

		num_ret = max(1, int(len(retained_clients) * self.online_rate_C))
		num_ret = min(len(retained_clients), num_ret)

		choose_indices = np.random.choice(len(retained_clients), num_ret, replace=False)
		selected_ret = [retained_clients[i] for i in choose_indices]

		if is_warmup:
			protocol_clients = selected_ret
		else:
			# 【核心修复 1】：终极万能判活逻辑，无视 Object 与 Int 类型的差异
			online_ids = [str(x.id) if hasattr(x, 'id') else str(x) for x in getattr(self, 'online_client_list', [])]
			surviving_unlearn_clients = [c for c in unlearn_clients if str(c.id) in online_ids]

			if len(unlearn_clients) > 0 and len(surviving_unlearn_clients) == 0:
				unlearn_ids = [c.id for c in unlearn_clients]
				print(
					f"⚠️ [Warning] Unlearn client (ID: {unlearn_ids}) dropped out in round {self.current_comm_round}! Unlearning paused for this round.")

			protocol_clients = selected_ret + surviving_unlearn_clients

		N_lsa = len(protocol_clients)
		if N_lsa == 0:
			return

		surviving_count = N_lsa
		U_lsa = surviving_count
		T_lsa = max(1, U_lsa // 2)
		if U_lsa <= T_lsa: U_lsa = T_lsa + 1

		W_lsa = self._generate_mds_matrix(U_lsa, N_lsa)

		is_secagg_param = self.params.get('use_secagg', True) if hasattr(self, 'params') and self.params else True
		if isinstance(is_secagg_param, str):
			is_secagg_enabled = is_secagg_param.lower() in ['true', '1', 't', 'y', 'yes']
		else:
			is_secagg_enabled = bool(is_secagg_param)

		beta_to_send = 0.0 if is_warmup else self.beta
		gamma_to_send = 0.0 if is_warmup else self.gamma

		Z_total_samples = sum([c.local_training_number for c in selected_ret])
		delta_total_samples = sum([c.local_training_number for c in protocol_clients])

		all_Z_shares = {j: {} for j in range(N_lsa)}
		all_delta_shares = {j: {} for j in range(N_lsa)}
		Z_ciphers, delta_ciphers = {}, {}
		Z_meta, delta_meta = None, None
		Z_plaintexts, delta_plaintexts = {}, {}

		do_update_U = True if is_warmup else (
					self.u_update_freq > 0 and (self.current_comm_round + 1) % self.u_update_freq == 0)

		_, _, raw_g_locals = self.train(target_client_list=protocol_clients)

		for idx, client in enumerate(protocol_clients):
			msg = {
				'command': 'cal_secagg_update',
				'raw_g_local': raw_g_locals[idx],
				'Z_total_samples': Z_total_samples,
				'delta_total_samples': delta_total_samples,
				'U_t': self.U if do_update_U else None,
				'beta': beta_to_send,
				'gamma': gamma_to_send,
				'do_projection': self.do_projection,
				'use_secagg': is_secagg_enabled,
				'U_lsa': U_lsa, 'T_lsa': T_lsa, 'N_lsa': N_lsa, 'W_lsa': W_lsa
			}
			res = client.get_message(msg)

			Z_ciphers[idx] = res.get('Z_i_cipher')
			delta_ciphers[idx] = res.get('delta_i_cipher')
			Z_plaintexts[idx] = res.get('plaintext_Z')
			delta_plaintexts[idx] = res.get('plaintext_delta')

			if is_secagg_enabled:
				if res.get('Z_shares'):
					for j in range(N_lsa): all_Z_shares[j][idx] = res['Z_shares'][j]
					Z_meta = res['Z_meta']
				for j in range(N_lsa): all_delta_shares[j][idx] = res['delta_shares'][j]
				delta_meta = res['delta_meta']

		surviving_indices = list(range(N_lsa))
		decoder_indices = surviving_indices[:U_lsa]
		Z_agg_shares, delta_agg_shares = [], []

		if is_secagg_enabled:
			for j in surviving_indices:
				protocol_clients[j].get_message(
					{'command': 'store_shares', 'Z_shares': all_Z_shares[j], 'delta_shares': all_delta_shares[j]})
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
					if Z_ciphers[i] is not None: Z_cipher_sum = torch.remainder(Z_cipher_sum + Z_ciphers[i],
					                                                            self.secagg_math.q)
				Z_sum_fq = torch.remainder(Z_cipher_sum - Z_mask_sum, self.secagg_math.q)
				Z_sum_real = self.secagg_math.dequantize_from_finite_field(Z_sum_fq, 1)

				true_Z_sum = sum([Z_plaintexts[i] for i in surviving_indices if Z_plaintexts.get(i) is not None])
				diff_Z = torch.max(torch.abs(Z_sum_real - true_Z_sum)).item()
				# 【探针 1 保留并精简输出】
				print(
					f"🕵️‍♂️ [探针] 第 {self.current_comm_round if not is_warmup else 'Warm-up'} 轮 | Z矩阵解密误差: {diff_Z:.8f}")

				self.update_subspace_secagg(Z_sum_real)

			if not is_warmup and delta_mask_sum is not None:
				delta_cipher_sum = torch.zeros_like(delta_mask_sum, dtype=torch.int64)
				for i in surviving_indices:
					delta_cipher_sum = torch.remainder(delta_cipher_sum + delta_ciphers[i], self.secagg_math.q)
				delta_sum_fq = torch.remainder(delta_cipher_sum - delta_mask_sum, self.secagg_math.q)
				avg_grad = self.secagg_math.dequantize_from_finite_field(delta_sum_fq, 1)

				true_delta_sum = sum([delta_plaintexts[i] for i in surviving_indices])
				diff_delta = torch.max(torch.abs(avg_grad - true_delta_sum)).item()
				# 【探针 2 保留并精简输出】
				print(f"🚀 [探针] 第 {self.current_comm_round} 轮 | 模型梯度解密误差: {diff_delta:.8f}")

				self.update_module(self.module, self.optimizer, self.lr, avg_grad)

		# --- 明文处理分支 ---
		else:
			if do_update_U:
				z_list = [Z_ciphers[i] for i in surviving_indices if Z_ciphers[i] is not None]
				if z_list:
					self.update_subspace_secagg(sum(z_list))

			if not is_warmup:
				grad_sum = sum([delta_ciphers[i] for i in surviving_indices])
				self.update_module(self.module, self.optimizer, self.lr, grad_sum)

		cal_time_end = time.time()
		if not hasattr(self, 'computation_time'): self.computation_time = 0.0
		self.computation_time += cal_time_end - cal_time_start

	def run(self):
		import os

		for client in self.client_list:
			if client.unlearn_flag:
				client.criterion = self.UnLearningCELoss()

		print("\n" + "=" * 60)
		print("🔍 [诊断] 检查预训练子空间继承状态...")

		if self.U is None:
			save_dir = os.path.join(os.getcwd(), 'saved_subspaces')

			# 获取核心参数，加入 NC 和 C
			seed = self.params.get('seed', 'unknown') if hasattr(self, 'params') else 'unknown'
			dataloader = self.params.get('dataloader', 'data') if hasattr(self, 'params') else 'data'
			module = self.params.get('module', 'model') if hasattr(self, 'params') else 'model'
			n_clients = self.params.get('N', self.client_num) if hasattr(self, 'params') else self.client_num
			nc = self.params.get('NC', 'all') if hasattr(self, 'params') else 'all'
			c_rate = self.params.get('C', 1.0) if hasattr(self, 'params') else 1.0

			# 👇 匹配带有 NC 和 C 的文件名
			file_name = f"U_seed{seed}_{dataloader}_{module}_N{n_clients}_NC{nc}_C{c_rate}_k{self.k}.pt"
			u_path = os.path.join(save_dir, file_name)

			print(f"🎯 正在寻找参数匹配的子空间文件: {file_name}")

			if os.path.exists(u_path):
				self.U = torch.load(u_path, map_location=self.device)
				print(f"✅ [INHERIT SUCCESS] 成功加载匹配的预训练子空间 U!")
				print(f"📁 文件绝对路径: {u_path}")
			else:
				print(f"⚠️ [INHERIT FAILED] 未找到匹配的预训练子空间文件!")
				print(f"📁 尝试寻找的路径: {u_path}")
				print(f"🔄 正在执行回退: 随机初始化子空间 U...")
				self.init_subspace()
		else:
			print(f"✅ 子空间 U 已在内存中存在，无需重新加载。")
		print("=" * 60 + "\n")

		print(f"=== Start SECURE Warm-up Phase ({self.warmup_rounds} rounds, Online Rate C={self.online_rate_C}) ===")
		for j in range(self.warmup_rounds):
			self.train_a_round(is_warmup=True)
			print(f"Secure Warm-up Round {j + 1}/{self.warmup_rounds} Done.")
		print("=== Warm-up Phase Finished. Subspace Ready. ===\n")

		print("=== Start SECURE Unlearning Phase ===")
		while not self.terminated():
			self.train_a_round(is_warmup=False)
