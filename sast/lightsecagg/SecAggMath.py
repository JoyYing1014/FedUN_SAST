import torch


class SecAggMath:
	def __init__(self, bit_length=55, scale=1e5):
		#  有限域的大小 q
		self.q = (1 << bit_length) - 1
		self.scale = scale   # 缩放因子，保留浮点数的小数精度

	def quantize_to_finite_field(self, tensor):
		"""将实数张量量化并映射到有限域 F_q"""
		tensor_int = torch.round(tensor * self.scale).to(torch.int64)
		# 映射到有限域 [0, q-1]，利用 Python 的模运算处理负数
		tensor_fq = torch.remainder(tensor_int, self.q)
		return tensor_fq

	def generate_mask_in_fq(self, shape, device):
		"""在有限域内生成均匀分布的随机整数掩码"""
		return torch.randint(0, self.q, shape, dtype=torch.int64, device=device)

	def dequantize_from_finite_field(self, tensor_fq, num_clients):
		"""将有限域聚合结果反量化为实数 (求平均)"""
		tensor_fq = tensor_fq.clone()
		half_q = self.q // 2

		# 恢复负数: 大于 q/2 的数在反量化时被视作负数
		is_negative = tensor_fq > half_q
		tensor_fq[is_negative] -= self.q

		# 缩小回浮点数，并除以参与人数得到平均值
		tensor_real = (tensor_fq.to(torch.float64) / self.scale / num_clients).to(torch.float32)
		return tensor_real
