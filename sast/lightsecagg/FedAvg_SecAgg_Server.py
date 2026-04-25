import sast as fs
import time
import torch
import numpy as np
from sast.lightsecagg.SecAggMath import SecAggMath


# 注意：这里继承的是底层框架原本的 FedAvg
class FedAvg_SecAgg_Server(fs.FedAvg):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.secagg_math = SecAggMath()

	def lightsecagg_decode_fq(self, cipher_list, mask_attr, clients):
		"""有限域密文解密 (与 FedUN_Server 中的解密逻辑完全一致)"""
		valid_ciphers = [c for c in cipher_list if c is not None]
		if not valid_ciphers:
			return None

		sum_ciphers = torch.zeros_like(valid_ciphers[0], dtype=torch.int64)
		for c in valid_ciphers:
			sum_ciphers = torch.remainder(sum_ciphers + c, self.secagg_math.q)

		sum_masks = torch.zeros_like(sum_ciphers, dtype=torch.int64)
		for client in clients:
			mask = getattr(client, mask_attr)
			if mask is not None:
				sum_masks = torch.remainder(sum_masks + mask, self.secagg_math.q)

		sum_fq = torch.remainder(sum_ciphers - sum_masks, self.secagg_math.q)
		return sum_fq

	def train_a_round(self):
		"""重写预训练的 train_a_round，使用全密态安全聚合"""
		com_time_start = time.time()
		cal_time_start = time.time()

		# 1. 预训练阶段的随机抽样 (所有客户端一视同仁，没有保留/遗忘之分)
		num_participate = max(1, int(self.client_num * self.sampling_rate))
		choose_indices = np.random.choice(self.client_num, num_participate, replace=False)
		current_clients = [self.client_list[i] for i in choose_indices]

		delta_i_cipher_list = []
		total_samples = 0

		# 2. 向客户端发送【密态指令】
		for client in current_clients:
			msg = {
				'command': 'cal_secagg_update',
				'U_t': None,  # 预训练阶段没有正交子空间
				'C_clip': 1e9,  # 预训练不执行 FedUN 裁剪，设为极大值
				'beta': 1.0,  # 预训练梯度权重为正常 1.0
				'gamma': 1.0,  # 预训练梯度权重为正常 1.0
				'do_projection': False,  # 预训练不执行遗忘投影
				'epochs': self.epochs,
				'lr': self.lr,
				'target_module': self.module
			}
			res = client.get_message(msg)

			# 因为 U_t 是 None，客户端本地的 Z_i 会是 None
			# 客户端只会返回加密后的常规梯度 delta_i_cipher
			delta_i_cipher_list.append(res['delta_i_cipher'])
			total_samples += res['n_i']

		com_time_end = time.time()

		# 3. 服务器在有限域上解密并更新全局模型
		if total_samples > 0:
			print(f"\n[安全聚合触发] FedAvg 预训练阶段: 服务器正在消除 {len(current_clients)} 个客户端的掩码...")

			# 密文抵消
			delta_sum_fq = self.lightsecagg_decode_fq(delta_i_cipher_list, 'mask_Delta_fq', current_clients)
			# 反量化恢复实数平均梯度
			avg_grad_real = self.secagg_math.dequantize_from_finite_field(delta_sum_fq, total_samples)

			# 正常更新全局模型权重
			self.update_module(self.module, self.optimizer, self.lr, avg_grad_real)

		cal_time_end = time.time()
		self.communication_time += com_time_end - com_time_start
		self.computation_time += cal_time_end - cal_time_start