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
		# 👇 提前计算 total_samples！
		total_samples = sum([c.local_training_number for c in protocol_clients])
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
		# 👇 探针变量：用于存放收到的明文
		Z_plaintexts, delta_plaintexts = {}, {}
		# 👆 =======================

		# =========================================================================
		# 【核心修复：分离预热与遗忘的子空间更新逻辑】
		# 预热阶段：每一轮都强制更新子空间 U
		# 遗忘阶段：受参数 u_update_freq 控制，为 0 时则完全不更新
		# =========================================================================
		if is_warmup:
			do_update_U = True
		else:
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
				'total_samples': total_samples,
				'use_secagg': is_secagg_enabled,
				'U_lsa': U_lsa, 'T_lsa': T_lsa, 'N_lsa': N_lsa, 'W_lsa': W_lsa
			}
			res = client.get_message(msg)
			Z_ciphers[idx] = res.get('Z_i_cipher')
			delta_ciphers[idx] = res.get('delta_i_cipher')
			# 👇 探针逻辑：提取明文
			Z_plaintexts[idx] = res.get('plaintext_Z')
			delta_plaintexts[idx] = res.get('plaintext_delta')
			# 👆 ===================

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
		# =========== 计算存活比例补偿 ===========
		# total_samples = sum([c.local_training_number for c in protocol_clients])
		surviving_p_sum = sum([protocol_clients[i].local_training_number for i in surviving_indices]) / total_samples

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

				# 🕵️‍♂️【探针验证 1：Z 矩阵】=========================
				true_Z_sum = sum([Z_plaintexts[i] for i in surviving_indices if Z_plaintexts.get(i) is not None])
				diff_Z = torch.max(torch.abs(Z_sum_real - true_Z_sum)).item()
				print(
					f"🕵️‍♂️ [探针] 第 {self.current_comm_round if not is_warmup else 'Warm-up'} 轮 - Z矩阵解密最大误差: {diff_Z:.8f}")
				assert diff_Z < 1e-3, f"Z矩阵安全聚合解密失败！误差过大: {diff_Z}"
				# ===============================================

				# 👇 加入掉线补偿！
				if surviving_p_sum > 0: Z_sum_real = Z_sum_real / surviving_p_sum
				self.update_subspace_secagg(Z_sum_real)

			# 【修复3】：严格保证只有在不是预热阶段时，才更新模型
			if not is_warmup and delta_mask_sum is not None:
				delta_cipher_sum = torch.zeros_like(delta_mask_sum, dtype=torch.int64)
				for i in surviving_indices:
					delta_cipher_sum = torch.remainder(delta_cipher_sum + delta_ciphers[i], self.secagg_math.q)
				delta_sum_fq = torch.remainder(delta_cipher_sum - delta_mask_sum, self.secagg_math.q)
				avg_grad = self.secagg_math.dequantize_from_finite_field(delta_sum_fq, 1)

				# 🕵️‍♂️【探针验证 2：模型参数梯度】====================
				true_delta_sum = sum([delta_plaintexts[i] for i in surviving_indices])
				diff_delta = torch.max(torch.abs(avg_grad - true_delta_sum)).item()
				print(f"🕵️‍♂️ [探针] 第 {self.current_comm_round} 轮 - 模型梯度解密最大误差: {diff_delta:.8f}")
				assert diff_delta < 1e-3, f"模型梯度安全聚合解密失败！误差过大: {diff_delta}"
				# ===============================================

				# 👇 加入掉线补偿！
				if surviving_p_sum > 0: avg_grad = avg_grad / surviving_p_sum
				self.update_module(self.module, self.optimizer, self.lr, avg_grad)

		# --- 明文处理分支 ---
		else:
			if do_update_U:
				z_list = [Z_ciphers[i] for i in surviving_indices if Z_ciphers[i] is not None]
				if z_list:
					Z_sum_real = sum(z_list)
					# 👇 加入掉线补偿！
					if surviving_p_sum > 0: Z_sum_real = Z_sum_real / surviving_p_sum
					self.update_subspace_secagg(Z_sum_real)

			if not is_warmup:
				grad_sum = sum([delta_ciphers[i] for i in surviving_indices])
				# 👇 加入掉线补偿！
				if surviving_p_sum > 0: grad_sum = grad_sum / surviving_p_sum
				self.update_module(self.module, self.optimizer, self.lr, grad_sum)

		cal_time_end = time.time()
		if not hasattr(self, 'computation_time'): self.computation_time = 0.0
		self.computation_time += cal_time_end - cal_time_start

	def run(self):
		import os

		# 翻转遗忘客户端损失函数，确保向背离后门方向优化
		for client in self.client_list:
			if client.unlearn_flag:
				client.criterion = self.UnLearningCELoss()

		# === 核心：自动继承预训练阶段维护的真实子空间 ===
		if self.U is None:
			u_path = os.path.join(getattr(self, 'save_dir', '.'), 'pretrained_U.pt')
			if os.path.exists(u_path):
				self.U = torch.load(u_path, map_location=self.device)
				print(f"\n=> [INHERIT SUCCESS] Successfully loaded true pre-trained subspace U from '{u_path}'!")
			else:
				self.init_subspace()

		# ---------------- 预热阶段 ----------------
		print(f"\n=== Start SECURE Warm-up Phase ({self.warmup_rounds} rounds, Online Rate C={self.online_rate_C}) ===")
		for j in range(self.warmup_rounds):
			self.train_a_round(is_warmup=True)
			print(f"Secure Warm-up Round {j + 1}/{self.warmup_rounds} Done.")
		print("=== Warm-up Phase Finished. Subspace Ready. ===\n")

		# ---------------- 遗忘阶段 ----------------
		print("=== Start SECURE Unlearning Phase ===")
		# 将测试和轮次推进依然交还给父类 while 循环底层的 update_module
		while not self.terminated():
			self.train_a_round(is_warmup=False)
