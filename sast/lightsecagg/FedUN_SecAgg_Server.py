import time
import torch
import numpy as np
from sast.algorithm.unlearning.FedUN import FedUN
from sast.lightsecagg.SecAggMath import SecAggMath


class FedUN_SecAgg_Server(FedUN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.secagg_math = SecAggMath()  # 注意：SecAggMath 中的 bit_length 应设为 61
		# 初始梯度裁剪阈值
		self.C_clip = self.params.get('C_clip', 5.0) if self.params else 5.0

	def lightsecagg_decode_fq(self, cipher_list, mask_attr, clients):
		"""有限域密文解密（模拟MDS消除掩码）"""
		valid_ciphers = [c for c in cipher_list if c is not None]
		if not valid_ciphers:
			return None

		# 1. 密文求和取模
		sum_ciphers = torch.zeros_like(valid_ciphers[0], dtype=torch.int64)
		for c in valid_ciphers:
			sum_ciphers = torch.remainder(sum_ciphers + c, self.secagg_math.q)

		# 2. 获取掩码总和
		sum_masks = torch.zeros_like(sum_ciphers, dtype=torch.int64)
		for client in clients:
			mask = getattr(client, mask_attr)
			if mask is not None:
				sum_masks = torch.remainder(sum_masks + mask, self.secagg_math.q)

		# 3. 密文和 - 掩码和 mod q
		sum_fq = torch.remainder(sum_ciphers - sum_masks, self.secagg_math.q)
		return sum_fq

	def update_subspace_secagg(self, Z_sum):
		"""直接使用解密后的 Z_sum 更新子空间"""
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
		"""重写 train_a_round，执行全密态安全聚合"""
		com_time_start = time.time()
		cal_time_start = time.time()

		unlearn_clients = [c for c in self.client_list if c.unlearn_flag]
		retained_clients = [c for c in self.client_list if not c.unlearn_flag]

		# 1. 抽样客户端
		current_clients = []
		if len(retained_clients) > 0:
			num_ret = max(1, int(len(retained_clients) * self.sampling_rate))
			num_ret = min(len(retained_clients), num_ret)
			choose_indices = np.random.choice(len(retained_clients), num_ret, replace=False)
			current_clients.extend([retained_clients[i] for i in choose_indices])

		current_clients.extend(unlearn_clients)  # 遗忘客户端强制参与

		do_update_U = (self.u_update_freq > 0 and (self.current_comm_round + 1) % self.u_update_freq == 0)
		U_t_to_send = self.U if do_update_U else None

		Z_i_cipher_list = []
		delta_i_cipher_list = []
		Z_clients = []

		# 【核心修改 4】：提前计算本轮参与客户端的总样本量
		total_samples = sum([client.local_training_number for client in current_clients])

		# 尝试从命令行参数读取开关，如果没有则默认开启
		is_secagg_enabled = getattr(self.params, 'use_secagg', True) if hasattr(self, 'params') else getattr(self, 'use_secagg', True)

		# 2. 收集密文
		for client in current_clients:
			msg = {
				'command': 'cal_secagg_update',
				'U_t': U_t_to_send,
				'C_clip': self.C_clip,
				'beta': self.beta,
				'gamma': self.gamma,
				'do_projection': self.do_projection,
				'epochs': self.epochs,
				'lr': self.lr,
				'target_module': self.module,
				'total_samples': total_samples,  # <--- 将计算好的总数传给客户端
				'use_secagg': is_secagg_enabled  # 👇 新增这一行：把指令下发给客户端
			}
			res = client.get_message(msg)

			if res.get('Z_i_cipher') is not None:
				Z_i_cipher_list.append(res['Z_i_cipher'])
				Z_clients.append(client)

			delta_i_cipher_list.append(res['delta_i_cipher'])

		com_time_end = time.time()

		# 3. 服务器安全解密与模型更新

		# 尝试从命令行参数读取开关，如果没有则默认开启 (确保容错)
		is_secagg_enabled = getattr(self.params, 'use_secagg', True) if hasattr(self, 'params') else getattr(self,
		                                                                                                     'use_secagg',
		                                                                                                     True)

		# (A) 解密/聚合更新子空间
		if do_update_U and len(Z_i_cipher_list) > 0:
			if is_secagg_enabled:
				# 【加密模式】：解码掩码并反量化
				Z_sum_fq = self.lightsecagg_decode_fq(Z_i_cipher_list, 'mask_Z_fq', Z_clients)
				# 反量化时直接除以1，因为我们要的是外积矩阵的总和
				Z_sum_real = self.secagg_math.dequantize_from_finite_field(Z_sum_fq, num_clients=1)
			else:
				# 【明文模式】：直接将客户端传来的真实浮点张量求和
				Z_sum_real = sum(Z_i_cipher_list)

			self.update_subspace_secagg(Z_sum_real)

		# (B) 解密/聚合更新模型
		if total_samples > 0 and len(delta_i_cipher_list) > 0:
			if is_secagg_enabled:
				# 【加密模式】：解码掩码并反量化
				delta_sum_fq = self.lightsecagg_decode_fq(delta_i_cipher_list, 'mask_Delta_fq', current_clients)
				# 【核心保留】：客户端已经用 p_i 做过比例加权，这里的总和就是标准的联邦平均梯度
				# 因此，反量化时直接除以 1 (或传入 num_clients=1)，不要再除以参与人数或样本总数！
				avg_grad_real = self.secagg_math.dequantize_from_finite_field(delta_sum_fq, num_clients=1)
			else:
				# 【明文模式】：客户端传回来的已经是加权后的真实梯度，直接求和即为联邦平均梯度
				avg_grad_real = sum(delta_i_cipher_list)

			self.update_module(self.module, self.optimizer, self.lr, avg_grad_real)

		cal_time_end = time.time()
		self.communication_time += com_time_end - com_time_start
		self.computation_time += cal_time_end - cal_time_start

	def run(self):
		"""重写 run，确保 Warm-up 阶段的子空间更新也处于全密态保护之下"""
		for client in self.client_list:
			if client.unlearn_flag:
				client.criterion = self.UnLearningCELoss()

		if self.U is None:
			self.init_subspace()

		print(f"=== Start SECURE Warm-up Phase for {self.warmup_rounds} rounds ===")
		retained_clients = [c for c in self.client_list if not c.unlearn_flag]

		for j in range(self.warmup_rounds):
			if len(retained_clients) > 0:
				num_participate = max(1, int(len(retained_clients) * self.sampling_rate))
				num_participate = min(len(retained_clients), num_participate)
				choose_indices = np.random.choice(len(retained_clients), num_participate, replace=False)
				current_clients = [retained_clients[i] for i in choose_indices]

				Z_i_cipher_list = []
				Z_clients = []

				# 预热阶段也要计算 total_samples 传下去（虽然这里 beta=0 没用到，但保持代码一致性）
				total_samples = sum([client.local_training_number for client in current_clients])

				for client in current_clients:
					msg = {
						'command': 'cal_secagg_update',
						'U_t': self.U,
						'C_clip': self.C_clip,
						'beta': 0.0,  # 预热阶段不更新模型
						'gamma': 0.0,  # 预热阶段不更新模型
						'do_projection': False,
						'epochs': self.epochs,
						'lr': self.lr,
						'target_module': self.module,
						'total_samples': total_samples
					}
					res = client.get_message(msg)
					if res.get('Z_i_cipher') is not None:
						Z_i_cipher_list.append(res['Z_i_cipher'])
						Z_clients.append(client)

				if len(Z_i_cipher_list) > 0:
					Z_sum_fq = self.lightsecagg_decode_fq(Z_i_cipher_list, 'mask_Z_fq', Z_clients)
					Z_sum_real = self.secagg_math.dequantize_from_finite_field(Z_sum_fq, num_clients=1)
					self.update_subspace_secagg(Z_sum_real)

			print(f"Secure Warm-up Round {j + 1}/{self.warmup_rounds} Done.")
			# 【你需要在这里插入以下 3 行代码】：
			self.current_comm_round = j  # 确保当前轮次正确
			self.test()  # 强制进行全量考试
			if hasattr(self, 'outFunc') and self.outFunc is not None:
				# 智能提取静态方法的底层函数
				actual_func = self.outFunc.__func__ if isinstance(self.outFunc, staticmethod) else self.outFunc
				actual_func(self)

		print("=== Start SECURE Unlearning Phase ===")
		while not self.terminated():
			self.train_a_round()
			# 👇 【在此处新增以下 3 行代码】👇
			self.current_comm_round += 1  # 确保轮数正常推进
			self.test()  # 1. 触发所有客户端考试，把分数写进日志字典
			if hasattr(self, 'outFunc') and self.outFunc is not None:
				# 智能提取静态方法的底层函数
				actual_func = self.outFunc.__func__ if isinstance(self.outFunc, staticmethod) else self.outFunc
				actual_func(self)