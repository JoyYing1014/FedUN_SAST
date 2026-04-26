import os
import time
import torch
import numpy as np
import random
import sast as fs
from sast.lightsecagg.SecAggMath import SecAggMath


class FedAvg_SecAgg_Server(fs.FedAvg):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.secagg_math = SecAggMath(bit_length=31, scale=1e5)
		self.C_clip = self.params.get('C_clip', 5.0) if hasattr(self, 'params') and self.params else 5.0
		self.online_rate_C = self.params.get('C', 1.0) if hasattr(self, 'params') and self.params else 1.0

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
		print(f"\n[FedAvg] Initializing pre-training subspace U with dimension k={self.k}...")
		dim = len(self.module.span_model_params_to_vec())
		rand_mat = torch.randn(dim, self.k).to(self.device)
		self.U, _ = torch.linalg.qr(rand_mat)

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

	def train_a_round(self):
		com_time_start = time.time()

		num_participate = max(1, int(self.client_num * self.online_rate_C))
		choose_indices = np.random.choice(self.client_num, num_participate, replace=False)
		current_clients = [self.client_list[i] for i in choose_indices]
		N_lsa = len(current_clients)

		# 取消二次掉线，因为采样率 C 在选人时已经生效了
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

		do_update_U = self.pretrain_subspace_update and ((self.current_comm_round + 1) % self.T_freq == 0)
		if do_update_U and self.U is None:
			self.init_subspace()

		all_delta_shares = {j: {} for j in range(N_lsa)}
		all_Z_shares = {j: {} for j in range(N_lsa)}
		delta_ciphers, Z_ciphers = {}, {}
		delta_plaintexts, Z_plaintexts = {}, {}
		delta_meta, Z_meta = None, None

		total_samples = sum([c.local_training_number for c in current_clients])

		# 【核心同步】：让原生框架跑出完美梯度，杜绝单 Batch 缩水现象！
		_, _, raw_g_locals = self.train(target_client_list=current_clients)

		for idx, client in enumerate(current_clients):
			msg = {
				'command': 'cal_secagg_update',
				'raw_g_local': raw_g_locals[idx],  # 【关键】发送原生明文梯度供客户端加密
				'Z_total_samples': total_samples,
				'delta_total_samples': total_samples,
				'U_t': self.U if do_update_U else None,
				'C_clip': self.C_clip,
				'beta': 1.0,
				'gamma': 1.0,
				'do_projection': False,
				'epochs': self.epochs,
				'lr': self.lr,
				'target_module': self.module,
				'use_secagg': is_secagg_enabled,
				'U_lsa': U_lsa, 'T_lsa': T_lsa, 'N_lsa': N_lsa, 'W_lsa': W_lsa
			}
			res = client.get_message(msg)

			delta_ciphers[idx] = res.get('delta_i_cipher')
			delta_plaintexts[idx] = res.get('plaintext_delta')

			if do_update_U:
				Z_ciphers[idx] = res.get('Z_i_cipher')
				Z_plaintexts[idx] = res.get('plaintext_Z')

			if is_secagg_enabled:
				if res.get('delta_shares'):
					for j in range(N_lsa): all_delta_shares[j][idx] = res['delta_shares'][j]
					delta_meta = res['delta_meta']
				if do_update_U and res.get('Z_shares'):
					for j in range(N_lsa): all_Z_shares[j][idx] = res['Z_shares'][j]
					Z_meta = res['Z_meta']

		surviving_indices = list(range(N_lsa))
		decoder_indices = surviving_indices[:U_lsa]
		delta_agg_shares, Z_agg_shares = [], []

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

		if is_secagg_enabled:
			delta_mask_sum = self.secagg_math.lightsecagg_decode(delta_agg_shares, decoder_indices, W_lsa, U_lsa, T_lsa,
			                                                     delta_meta[0],
			                                                     delta_meta[1]) if delta_agg_shares else None

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

					true_Z_sum = sum([Z_plaintexts[i] for i in surviving_indices if Z_plaintexts.get(i) is not None])
					diff_Z = torch.max(torch.abs(Z_sum_real - true_Z_sum)).item()
					print(f"=========================================================")
					print(f"🕵️‍♂️ [预训练安全探针] 第 {self.current_comm_round} 轮 | Z矩阵误差: {diff_Z:.8f}")
					print(f"=========================================================")
					self.update_subspace_secagg(Z_sum_real)

			if delta_mask_sum is not None:
				delta_cipher_sum = torch.zeros_like(delta_mask_sum, dtype=torch.int64)
				for i in surviving_indices:
					delta_cipher_sum = torch.remainder(delta_cipher_sum + delta_ciphers[i], self.secagg_math.q)
				delta_sum_fq = torch.remainder(delta_cipher_sum - delta_mask_sum, self.secagg_math.q)
				avg_grad = self.secagg_math.dequantize_from_finite_field(delta_sum_fq, 1)

				true_delta_sum = sum(
					[delta_plaintexts[i] for i in surviving_indices if delta_plaintexts.get(i) is not None])
				diff_delta = torch.max(torch.abs(avg_grad - true_delta_sum)).item()
				print(f"=========================================================")
				print(f"🚀 [预训练安全探针] 第 {self.current_comm_round} 轮 | 模型梯度误差: {diff_delta:.8f}")
				print(f"=========================================================")

				if surviving_p_sum > 0: avg_grad = avg_grad / surviving_p_sum
				self.update_module(self.module, self.optimizer, self.lr, avg_grad)
		else:
			if do_update_U:
				z_list = [Z_ciphers[i] for i in surviving_indices if Z_ciphers.get(i) is not None]
				if z_list:
					Z_sum_real = sum(z_list)
					self.update_subspace_secagg(Z_sum_real)

			grad_sum = sum([delta_ciphers[i] for i in surviving_indices])
			if surviving_p_sum > 0: grad_sum = grad_sum / surviving_p_sum
			self.update_module(self.module, self.optimizer, self.lr, grad_sum)

		cal_time_end = time.time()
		if not hasattr(self, 'computation_time'): self.computation_time = 0.0
		self.computation_time += cal_time_end - cal_time_start

	# 【核心修复】：霸道接管原生框架的控制权，强制执行 SecAgg 流程！
	def run(self):
		print("\n=== Start SECURE Pre-Training Phase ===")
		self.current_comm_round = 0
		while not self.terminated():
			self.train_a_round()

		# 预训练结束后保存子空间
		if getattr(self, 'pretrain_subspace_update', False) and self.U is not None:
			save_dir = os.path.join(os.getcwd(), 'saved_subspaces')
			os.makedirs(save_dir, exist_ok=True)

			seed = self.params.get('seed', 'unknown') if hasattr(self, 'params') else 'unknown'
			dataloader = self.params.get('dataloader', 'data') if hasattr(self, 'params') else 'data'
			module = self.params.get('module', 'model') if hasattr(self, 'params') else 'model'
			n_clients = self.params.get('N', self.client_num) if hasattr(self, 'params') else self.client_num
			nc = self.params.get('NC', 'all') if hasattr(self, 'params') else 'all'
			c_rate = self.params.get('C', 1.0) if hasattr(self, 'params') else 1.0

			file_name = f"U_seed{seed}_{dataloader}_{module}_N{n_clients}_NC{nc}_C{c_rate}_k{self.k}.pt"
			save_path = os.path.join(save_dir, file_name)

			torch.save(self.U, save_path)
			print("\n" + "=" * 60)
			print(f"✅ [SAVE SUCCESS] 预训练子空间 U 已成功保存！")
			print(f"📁 保存路径: {save_path}")
			print(
				f"🔑 绑定参数: Seed={seed}, Data={dataloader}, Model={module}, N={n_clients}, NC={nc}, C={c_rate}, k={self.k}")
			print("=" * 60 + "\n")
