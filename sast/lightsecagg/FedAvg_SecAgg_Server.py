import os
import time
import torch
import numpy as np
import random
import sast as fs
from sast.lightsecagg.SecAggMath import SecAggMath


# 注意：这里继承的是底层框架原本的 FedAvg
class FedAvg_SecAgg_Server(fs.FedAvg):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# 强制与遗忘阶段保持一致的安全位宽与缩放精度
		self.secagg_math = SecAggMath(bit_length=31, scale=1e5)
		self.C_clip = self.params.get('C_clip', 5.0) if hasattr(self, 'params') and self.params else 5.0
		self.online_rate_C = self.params.get('C', 1.0) if hasattr(self, 'params') and self.params else 1.0

		# === 新增：预训练子空间更新参数 ===
		self.pretrain_subspace_update = self.params.get('pretrain_subspace_update', False) if hasattr(self,
		                                                                                              'params') and self.params else False
		if isinstance(self.pretrain_subspace_update, str):
			self.pretrain_subspace_update = self.pretrain_subspace_update.lower() in ['true', '1', 't', 'y', 'yes']
		else:
			self.pretrain_subspace_update = bool(self.pretrain_subspace_update)

		self.T_freq = int(self.params.get('T_freq', 10)) if hasattr(self, 'params') and self.params else 10
		self.k = int(self.params.get('k', 20)) if hasattr(self, 'params') and self.params else 20
		self.rho = float(self.params.get('rho', 0.5)) if hasattr(self, 'params') and self.params else 0.5

		self.U = None

	def _generate_mds_matrix(self, U, N):
		W = torch.zeros((U, N), dtype=torch.int64, device=self.device)
		for i in range(U):
			for j in range(N):
				W[i, j] = pow(j + 1, i, self.secagg_math.q)
		return W

	def init_subspace(self):
		"""预训练阶段的初始随机子空间"""
		print(f"\n[FedAvg] Initializing pre-training subspace U with dimension k={self.k}...")
		dim = len(self.module.span_model_params_to_vec())
		rand_mat = torch.randn(dim, self.k).to(self.device)
		self.U, _ = torch.linalg.qr(rand_mat)

	def update_subspace_secagg(self, Z_sum):
		"""预训练阶段对齐并更新真实的子空间流形"""
		if Z_sum is None:
			return
		U_hat, _ = torch.linalg.qr(Z_sum)
		Q = self.U.T @ U_hat
		A, S, B_t = torch.linalg.svd(Q)
		R = A @ B_t
		U_hat_aligned = U_hat @ R
		U_next_unorth = (1 - self.rho) * self.U + self.rho * U_hat_aligned
		self.U, _ = torch.linalg.qr(U_next_unorth)

	def train_a_round(self):
		"""重写预训练的 train_a_round，执行 LightSecAgg，并按 T_freq 更新子空间"""
		com_time_start = time.time()

		# 1. 预训练阶段的随机抽样 (所有客户端一视同仁)
		num_participate = max(1, int(self.client_num * self.sampling_rate))
		choose_indices = np.random.choice(self.client_num, num_participate, replace=False)
		current_clients = [self.client_list[i] for i in choose_indices]
		N_lsa = len(current_clients)

		# 2. 掉线参数与 MDS 矩阵准备
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

		# === 核心逻辑：判断本轮是否需要更新子空间 ===
		do_update_U = self.pretrain_subspace_update and ((self.current_comm_round + 1) % self.T_freq == 0)
		if do_update_U and self.U is None:
			self.init_subspace()

		all_delta_shares = {j: {} for j in range(N_lsa)}
		all_Z_shares = {j: {} for j in range(N_lsa)}
		delta_ciphers, Z_ciphers = {}, {}
		# 👇 探针变量：用于存放收到的明文
		delta_plaintexts, Z_plaintexts = {}, {}
		# 👆 =======================
		delta_meta, Z_meta = None, None
		total_samples = sum([c.local_training_number for c in current_clients])

		# 3. 客户端执行本地训练、加密并生成碎片
		for idx, client in enumerate(current_clients):
			msg = {
				'command': 'cal_secagg_update',
				'U_t': self.U if do_update_U else None,  # 仅在需要更新的轮次下发 U_t
				'C_clip': self.C_clip,
				'beta': 1.0,  # 预训练正常梯度下降
				'gamma': 1.0,
				'do_projection': False,  # 预训练不执行遗忘投影
				'epochs': self.epochs,
				'lr': self.lr,
				'target_module': self.module,
				'total_samples': total_samples,
				'use_secagg': is_secagg_enabled,
				'U_lsa': U_lsa, 'T_lsa': T_lsa, 'N_lsa': N_lsa, 'W_lsa': W_lsa
			}
			res = client.get_message(msg)

			delta_ciphers[idx] = res.get('delta_i_cipher')
			# 👇 提取模型梯度明文
			delta_plaintexts[idx] = res.get('plaintext_delta')
			# 👆 =======================
			if do_update_U:
				Z_ciphers[idx] = res.get('Z_i_cipher')
				# 👇 提取 Z 矩阵明文
				Z_plaintexts[idx] = res.get('plaintext_Z')
				# 👆 =======================

			if is_secagg_enabled:
				if res.get('delta_shares'):
					for j in range(N_lsa): all_delta_shares[j][idx] = res['delta_shares'][j]
					delta_meta = res['delta_meta']

				# 同步处理 Z 矩阵的密文碎片
				if do_update_U and res.get('Z_shares'):
					for j in range(N_lsa): all_Z_shares[j][idx] = res['Z_shares'][j]
					Z_meta = res['Z_meta']

		# 4. 模拟掉线 (Dropouts)
		surviving_indices = random.sample(range(N_lsa), surviving_count)
		decoder_indices = surviving_indices[:U_lsa]
		delta_agg_shares, Z_agg_shares = [], []

		# 5. 路由下发与碎片聚合
		if is_secagg_enabled:
			for j in surviving_indices:
				current_clients[j].get_message({
					'command': 'store_shares',
					'Z_shares': all_Z_shares[j] if do_update_U else {},
					'delta_shares': all_delta_shares[j]
				})

			for j in decoder_indices:
				res = current_clients[j].get_message({
					'command': 'aggregate_shares',
					'surviving_indices': surviving_indices
				})
				if res.get('delta_agg') is not None: delta_agg_shares.append(res['delta_agg'])
				if do_update_U and res.get('Z_agg') is not None: Z_agg_shares.append(res['Z_agg'])

		com_time_end = time.time()
		if not hasattr(self, 'communication_time'): self.communication_time = 0.0
		self.communication_time += com_time_end - com_time_start

		cal_time_start = time.time()
		surviving_p_sum = sum([current_clients[i].local_training_number for i in surviving_indices]) / total_samples

		# 6. 一键解码与模型更新
		if is_secagg_enabled:
			delta_mask_sum = self.secagg_math.lightsecagg_decode(delta_agg_shares, decoder_indices, W_lsa, U_lsa, T_lsa,
			                                                     delta_meta[0],
			                                                     delta_meta[1]) if delta_agg_shares else None

			# --- 密态解密并更新子空间 Z ---
			if do_update_U:
				Z_mask_sum = self.secagg_math.lightsecagg_decode(Z_agg_shares, decoder_indices, W_lsa, U_lsa, T_lsa,
				                                                 Z_meta[0], Z_meta[1]) if Z_agg_shares else None
				if Z_mask_sum is not None:
					Z_cipher_sum = torch.zeros_like(Z_mask_sum, dtype=torch.int64)
					for i in surviving_indices:
						if Z_ciphers[i] is not None:
							Z_cipher_sum = torch.remainder(Z_cipher_sum + Z_ciphers[i], self.secagg_math.q)
					Z_sum_fq = torch.remainder(Z_cipher_sum - Z_mask_sum, self.secagg_math.q)
					Z_sum_real = self.secagg_math.dequantize_from_finite_field(Z_sum_fq, 1)
					# 🕵️‍♂️【探针 1：预训练 Z 矩阵】=========================
					true_Z_sum = sum([Z_plaintexts[i] for i in surviving_indices if Z_plaintexts.get(i) is not None])
					diff_Z = torch.max(torch.abs(Z_sum_real - true_Z_sum)).item()
					print(f"🕵️‍♂️ [预训练探针] 第 {self.current_comm_round} 轮 - Z矩阵最大误差: {diff_Z:.8f}")
					assert diff_Z < 1e-3, f"预训练 Z 矩阵安全聚合解密失败！误差: {diff_Z}"
					# ===============================================
					if surviving_p_sum > 0: Z_sum_real = Z_sum_real / surviving_p_sum
					self.update_subspace_secagg(Z_sum_real)

			# --- 密态更新全局模型 ---
			if delta_mask_sum is not None:
				delta_cipher_sum = torch.zeros_like(delta_mask_sum, dtype=torch.int64)
				for i in surviving_indices:
					delta_cipher_sum = torch.remainder(delta_cipher_sum + delta_ciphers[i], self.secagg_math.q)
				delta_sum_fq = torch.remainder(delta_cipher_sum - delta_mask_sum, self.secagg_math.q)
				avg_grad = self.secagg_math.dequantize_from_finite_field(delta_sum_fq, 1)
				# 🕵️‍♂️【探针 2：预训练全局模型梯度】====================
				true_delta_sum = sum(
					[delta_plaintexts[i] for i in surviving_indices if delta_plaintexts.get(i) is not None])
				diff_delta = torch.max(torch.abs(avg_grad - true_delta_sum)).item()
				print(f"🕵️‍♂️ [预训练探针] 第 {self.current_comm_round} 轮 - 模型梯度最大误差: {diff_delta:.8f}")
				assert diff_delta < 1e-3, f"预训练模型梯度安全聚合解密失败！误差: {diff_delta}"
				# ===============================================
				if surviving_p_sum > 0: avg_grad = avg_grad / surviving_p_sum
				self.update_module(self.module, self.optimizer, self.lr, avg_grad)
		else:
			# 明文回退分支
			if do_update_U:
				z_list = [Z_ciphers[i] for i in surviving_indices if Z_ciphers.get(i) is not None]
				if z_list:
					Z_sum_real = sum(z_list)
					if surviving_p_sum > 0: Z_sum_real = Z_sum_real / surviving_p_sum
					self.update_subspace_secagg(Z_sum_real)

			grad_sum = sum([delta_ciphers[i] for i in surviving_indices])
			if surviving_p_sum > 0: grad_sum = grad_sum / surviving_p_sum
			self.update_module(self.module, self.optimizer, self.lr, grad_sum)

		cal_time_end = time.time()
		if not hasattr(self, 'computation_time'): self.computation_time = 0.0
		self.computation_time += cal_time_end - cal_time_start

	def run(self):
		super().run()  # 执行父类原生的 FedAvg 训练流程

		# 预训练结束后，将更新好的真实子空间保存，供遗忘阶段无缝继承
		if self.pretrain_subspace_update and self.U is not None:
			save_path = os.path.join(getattr(self, 'save_dir', '.'), 'pretrained_U.pt')
			torch.save(self.U, save_path)
			print(f"\n=> [SAVE SUCCESS] Pre-trained subspace U saved to '{save_path}' for FedUN Warm-up inheritance.")
