import math
import torch


class SecAggMath:
	# 31位梅森素数 + 1e5缩放。
	# scale=1e5 保证在梯度爆炸到极端大小时也绝不溢出 q/2 (约10.7亿)
	# 同时使用随机量化 (Stochastic Rounding) 保证极小梯度无偏保留
	def __init__(self, bit_length=31, scale=1e5):
		self.q = (1 << bit_length) - 1
		self.scale = scale

	def quantize_to_finite_field(self, tensor):
		scaled_tensor = tensor * self.scale
		floor_tensor = torch.floor(scaled_tensor)
		prob = scaled_tensor - floor_tensor
		# 随机取整：按小数部分的概率向上取整，彻底杜绝小梯度被四舍五入归零
		tensor_int = floor_tensor + torch.bernoulli(prob).to(tensor.device)
		tensor_fq = torch.remainder(tensor_int.to(torch.int64), self.q)
		return tensor_fq

	def generate_mask_in_fq(self, shape, device):
		return torch.randint(0, self.q, shape, dtype=torch.int64, device=device)

	def dequantize_from_finite_field(self, tensor_fq, num_clients):
		tensor_fq = tensor_fq.clone()
		half_q = self.q // 2
		is_negative = tensor_fq > half_q
		tensor_fq[is_negative] -= self.q
		tensor_real = (tensor_fq.to(torch.float64) / self.scale / num_clients).to(torch.float32)
		return tensor_real

	def matrix_inverse_fq(self, matrix_tensor):
		q = self.q
		n = matrix_tensor.shape[0]
		mat = matrix_tensor.cpu().tolist()

		for i in range(n):
			mat[i].extend([1 if i == j else 0 for j in range(n)])

		for i in range(n):
			pivot_row = i
			while pivot_row < n and mat[pivot_row][i] % q == 0:
				pivot_row += 1
			if pivot_row == n:
				raise ValueError("Matrix is singular over F_q")
			if pivot_row != i:
				mat[i], mat[pivot_row] = mat[pivot_row], mat[i]

			pivot_val = mat[i][i] % q
			pivot_inv = pow(pivot_val, q - 2, q)
			for j in range(2 * n):
				mat[i][j] = (mat[i][j] * pivot_inv) % q

			for j in range(n):
				if j != i:
					factor = mat[j][i] % q
					if factor != 0:
						for k in range(2 * n):
							mat[j][k] = (mat[j][k] - factor * mat[i][k]) % q

		inv_mat = [row[n:] for row in mat]
		return torch.tensor(inv_mat, dtype=torch.int64, device=matrix_tensor.device)

	def lightsecagg_encode(self, tensor_fq, U, T, N, W, device):
		shape = tensor_fq.shape
		tensor_1d = tensor_fq.flatten()
		L = tensor_1d.numel()

		chunk_size = math.ceil(L / (U - T))
		pad_len = chunk_size * (U - T) - L

		if pad_len > 0:
			pad_tensor = torch.zeros(pad_len, dtype=torch.int64, device=device)
			z_padded = torch.cat([tensor_1d, pad_tensor])
		else:
			z_padded = tensor_1d

		z_matrix = z_padded.view(U - T, chunk_size)
		noise_matrix = self.generate_mask_in_fq((T, chunk_size), device)
		M = torch.cat([z_matrix, noise_matrix], dim=0)

		shares = {}
		for j in range(N):
			w_col = W[:, j]
			share_j = torch.zeros(chunk_size, dtype=torch.int64, device=device)
			for k in range(U):
				term = torch.remainder(w_col[k] * M[k], self.q)
				share_j = torch.remainder(share_j + term, self.q)
			shares[j] = share_j

		return shares, shape, pad_len

	def lightsecagg_decode(self, aggregated_shares, decoder_indices, W, U, T, original_shape, pad_len):
		S = torch.stack(aggregated_shares, dim=0)
		W_sub = W[:, decoder_indices]
		W_inv = self.matrix_inverse_fq(W_sub.T)

		M_sum = torch.zeros((U, S.shape[1]), dtype=torch.int64, device=S.device)
		for i in range(U):
			for k in range(U):
				term = torch.remainder(W_inv[i, k] * S[k], self.q)
				M_sum[i] = torch.remainder(M_sum[i] + term, self.q)

		z_sum_blocks = M_sum[:U - T, :]
		z_sum_1d = z_sum_blocks.flatten()

		if pad_len > 0:
			z_sum_1d = z_sum_1d[:-pad_len]

		return z_sum_1d.view(original_shape)
