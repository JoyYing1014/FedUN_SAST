import sast as fs
import torch
from sast.lightsecagg.SecAggMath import SecAggMath


class FedUN_SecAgg_Client(fs.Client):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.secagg_math = SecAggMath()

		# 记录每轮 LightSecAgg 的真实整数掩码
		self.mask_Z_fq = None
		self.mask_Delta_fq = None

	def get_message(self, msg):
		return_msg = {}

		# 拦截我们自定义的安全聚合命令
		if msg['command'] == 'cal_secagg_update':
			U_t = msg.get('U_t', None)
			C_clip = msg.get('C_clip', 1.0)
			beta = msg.get('beta', 1.0)
			gamma = msg.get('gamma', 5.0)
			do_projection = msg.get('do_projection', True)
			target_module = msg['target_module']

			# 1. 本地正常前向/反向传播，获取浮点数真实梯度 g_local
			if self.step_type == 'sgd':
				self.cal_gradient_loss_sgd(msg['epochs'], msg['lr'], target_module)
			else:
				self.cal_gradient_loss(msg['epochs'], msg['lr'], target_module)

			g_local = self.upload_grad
			n_i = self.local_training_number

			# 【核心修改 1】：获取服务器广播的总样本量，计算数据比例 p_i
			total_samples = msg.get('total_samples', n_i)
			p_i = float(n_i) / float(total_samples)

			Z_i = None
			delta_i = None

			# 2. 身份判断与本地计算
			if not self.unlearn_flag:
				# ====== 保留客户端 ======
				# a. 梯度裁剪
				norm_g = torch.norm(g_local)
				clip_factor = min(1.0, float(C_clip) / (float(norm_g) + 1e-6))
				g_bar = g_local * clip_factor

				# b. 构造贡献矩阵 Z_i (Z_i 是正交基底，不参与数据量加权)
				if U_t is not None:
					c_i = g_bar @ U_t
					Z_i = torch.outer(g_bar, c_i)

				# c. 最终模型更新量 (【核心修改 2】：使用比例 p_i 加权，避免溢出！)
				delta_i = p_i * beta * g_local
			else:
				# ====== 遗忘客户端 ======
				if U_t is not None and do_projection:
					# 本地执行正交投影，不向服务器泄露 g_u
					coeff = U_t.T @ g_local
					proj = U_t @ coeff
					r = g_local - proj

					if torch.norm(r) > 1e-6:
						scale = torch.norm(g_local) / torch.norm(r)
						d_u = scale * r
					else:
						d_u = torch.zeros_like(g_local)
				else:
					d_u = g_local

				# 【核心修改 3】：使用比例 p_i 加权遗忘梯度
				delta_i = p_i * gamma * d_u

			# 3. ====== LightSecAgg 核心加解密阶段 ======
			# 获取服务器下发的加密开关，默认开启
			use_secagg = msg.get('use_secagg', True)

			if use_secagg:
				# (A) 对 Z_i 加密 (使用量化和有限域掩码)
				Z_i_cipher = None
				if Z_i is not None:
					Z_i_fq = self.secagg_math.quantize_to_finite_field(Z_i)
					self.mask_Z_fq = self.secagg_math.generate_mask_in_fq(Z_i_fq.shape, self.device)
					Z_i_cipher = torch.remainder(Z_i_fq + self.mask_Z_fq, self.secagg_math.q)

				# (B) 对 模型更新量 delta_i 加密
				delta_i_fq = self.secagg_math.quantize_to_finite_field(delta_i)
				self.mask_Delta_fq = self.secagg_math.generate_mask_in_fq(delta_i_fq.shape, self.device)
				delta_i_cipher = torch.remainder(delta_i_fq + self.mask_Delta_fq, self.secagg_math.q)
			else:
				# 【明文模式】：直接原样返回真实浮点数张量（为了代码兼容，依然借用 _cipher 这个字典键名）
				Z_i_cipher = Z_i
				delta_i_cipher = delta_i

			# 仅返回密文(或明文)和基本信息
			return_msg['Z_i_cipher'] = Z_i_cipher
			return_msg['delta_i_cipher'] = delta_i_cipher
			return_msg['n_i'] = n_i
			return return_msg

		return super().get_message(msg)