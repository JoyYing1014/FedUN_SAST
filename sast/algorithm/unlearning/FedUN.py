import sast as fs
import time
import torch
import numpy as np
import torch.nn.functional as F


class FedUN(fs.UnlearnAlgorithm):
	def __init__(self,
	             name='FedUN',
	             data_loader=None,
	             module=None,
	             device=None,
	             train_setting=None,
	             client_num=None,
	             client_list=None,
	             online_client_num=None,
	             save_model=False,
	             max_comm_round=0,
	             max_training_num=0,
	             epochs=1,
	             save_name=None,
	             outFunc=None,
	             write_log=True,
	             dishonest=None,
	             test_conflicts=False,
	             params=None,
	             *args,
	             **kwargs):

		super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num,
		                 save_model, max_comm_round,
		                 max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
		# 强制将遗忘截止轮次设定为总轮次，防止底层框架在 UR 轮后踢掉遗忘客户端。
		self.UR = self.max_comm_round
		if hasattr(self, 'params') and self.params is not None:
			self.params['UR'] = self.max_comm_round
		# === 读取超参数 ===
		self.k = params.get('k', 20) if params else 20
		self.warmup_rounds = params.get('warmup_rounds', 10) if params else 10
		self.rho = params.get('rho', 0.5) if params else 0.5

		# 权重参数
		self.beta = params.get('sast_beta', 1.0) if params else 1.0
		self.gamma = params.get('sast_gamma', 5.0) if params else 5.0

		# 客户端抽取率
		self.sampling_rate = params.get('C', 1.0) if params else 1.0

		# === 新增：子空间在线更新频率 ===
		# 1 表示每轮更新，5 表示每5轮更新一次，0 表示不更新
		self.u_update_freq = params.get('u_update_freq', 1) if params else 1

		# 投影开关
		proj_param = params.get('do_projection', True)
		if isinstance(proj_param, str):
			self.do_projection = (proj_param.lower() == 'true')
		else:
			self.do_projection = bool(proj_param)

		self.U = None
		print(f"FedUN Settings: k={self.k}, beta={self.beta}, gamma={self.gamma}, "
		      f"Projection={self.do_projection}, Update Freq={self.u_update_freq}")

	def init_subspace(self):
		"""
		初始化子空间 U0: D x k 随机正交矩阵
		"""
		print(f"Initializing subspace U with dimension k={self.k}...")
		dim = len(self.module.span_model_params_to_vec())
		# 生成随机高斯矩阵
		rand_mat = torch.randn(dim, self.k).to(self.device)
		# QR分解获取正交基
		self.U, _ = torch.linalg.qr(rand_mat)

	# 【修改1】：传入对应的 participating_clients 以获取各自的样本量
	def update_subspace(self, g_locals, participating_clients):
		"""
		子空间更新逻辑 (Warm-up & Online Update)
		包含 Procrustes 对齐和平滑更新，并加入了样本量加权补偿
		"""
		if len(g_locals) == 0 or len(participating_clients) == 0:
			return

		# 1. 聚合协方差代理矩阵 Z
		g_stack = torch.stack(g_locals) if isinstance(g_locals, list) else g_locals

		# 计算投影系数 c = g^T * U_t  (N, k)
		c = g_stack @ self.U

		# === 加入基于样本量的加权补偿 ===
		total_samples = sum([client.local_training_number for client in participating_clients])
		if total_samples > 0:
			# 提取每个客户端的样本量比例，形状转换为 (N, 1) 以便在特征维度广播
			weights = torch.tensor(
				[client.local_training_number / total_samples for client in participating_clients],
				dtype=torch.float32, device=self.device
			).unsqueeze(1)

			# 使用比例对 g_stack 进行加权: g_i = g_i * (n_i / sum(n_i))
			g_stack_weighted = g_stack * weights
			# 构造补偿后的严格无偏贡献矩阵 Z = G_weighted.T @ C  (D, k)
			Z = g_stack_weighted.T @ c
		else:
			Z = g_stack.T @ c

		# 2. 计算候选基 U_hat_{t+1} = QR(Z)
		U_hat, _ = torch.linalg.qr(Z)

		# 3. 计算最佳旋转矩阵 R (Procrustes Alignment)
		# 目标: 寻找 R 使得 U_hat @ R 最接近 U_t
		Q = self.U.T @ U_hat

		# SVD 分解: Q = A * S * B^T
		A, S, B_t = torch.linalg.svd(Q)

		# 最佳旋转矩阵 R = A @ B^T
		R = A @ B_t

		# 4. 对齐
		U_hat_aligned = U_hat @ R

		# 5. 平滑更新 (EMA on Manifold)
		U_next_unorth = (1 - self.rho) * self.U + self.rho * U_hat_aligned
		self.U, _ = torch.linalg.qr(U_next_unorth)

	def get_projected_gradient(self, g_u):
		"""
		计算投影并缩放后的遗忘梯度
		proj_g = scale * (I - U U^T) g_u
		"""
		# 计算 g_u 在 U 上的投影 p = U (U^T g_u)
		coeff = self.U.T @ g_u
		proj = self.U @ coeff

		# 计算残差 (正交补分量) r = g_u - p
		r = g_u - proj

		norm_r = torch.norm(r)
		norm_g = torch.norm(g_u)

		# 数值稳定性处理
		epsilon = 1e-6
		if norm_r < epsilon:
			# 如果残差极小，说明梯度几乎完全在保留子空间内。
			proj_g = torch.zeros_like(g_u)
		else:
			# 缩放因子，保持模长与原梯度一致
			scale = norm_g / norm_r
			proj_g = scale * r

		return proj_g

	def train_a_round(self):
		"""
		执行一轮遗忘更新
		"""
		com_time_start = time.time()
		cal_time_start = time.time()

		unlearn_clients = [c for c in self.client_list if c.unlearn_flag]
		retained_clients = [c for c in self.client_list if not c.unlearn_flag]

		# 1. 抽取保留客户端
		# 注意：这里抽取的客户端不仅用于计算 Loss (Beta)，也用于更新子空间 U
		current_retained_clients = []
		if len(retained_clients) > 0:
			num_ret_participate = int(len(retained_clients) * self.sampling_rate)
			num_ret_participate = max(1, num_ret_participate)
			num_ret_participate = min(len(retained_clients), num_ret_participate)

			choose_indices = np.random.choice(len(retained_clients), num_ret_participate, replace=False)
			current_retained_clients = [retained_clients[i] for i in choose_indices]

		# 2. 收集梯度
		weighted_grads = []
		total_samples = 0
		g_ret_list = []  # 用于子空间更新

		# (a) 保留客户端：梯度下降
		if len(current_retained_clients) > 0:
			_, _, g_ret_list = self.train(target_client_list=current_retained_clients)

			for i, client in enumerate(current_retained_clients):
				n_i = client.local_training_number
				g_i = g_ret_list[i]

				# 如果 Beta > 0，则将保留梯度加入更新
				if self.beta > 0:
					weighted_grads.append(n_i * self.beta * g_i)
					total_samples += n_i

		# === 在线更新子空间 ===
		if self.u_update_freq > 0 and \
				(self.current_comm_round + 1) % self.u_update_freq == 0 and \
				len(g_ret_list) > 0:
			# 【修改2】：将当前的 current_retained_clients 传给 update_subspace 供其提取样本比例
			self.update_subspace(g_ret_list, current_retained_clients)

		# (b) 遗忘客户端：梯度上升
		# 必须先筛选出本轮【真正存活下来】的遗忘客户端
		surviving_unlearn_clients = [
			c for c in unlearn_clients
			if c in self.online_client_list
		]

		# 这里可以注释掉 print 防止刷屏
		# print("surviving_unlearn_clients:", surviving_unlearn_clients)

		if len(surviving_unlearn_clients) > 0:
			# 只有当遗忘客户端本轮没掉线，才计算遗忘梯度
			_, _, g_u_list = self.train(target_client_list=surviving_unlearn_clients)
			for i, client in enumerate(surviving_unlearn_clients):
				n_i = client.local_training_number
				g_u = g_u_list[i]

				# 投影处理 (使用可能刚刚更新过的 U)
				if self.do_projection:
					g_final = self.get_projected_gradient(g_u)
				else:
					g_final = g_u

				weighted_grads.append(n_i * self.gamma * g_final)
				total_samples += n_i
		else:
			# ⚠️ 遗忘客户端本轮掉线了！
			# 系统不崩溃，本轮只做保留客户端的恢复训练，暂停遗忘动作
			print(
				f"⚠️ [Warning] Unlearn client dropped out in round {self.current_comm_round}! Unlearning paused for this round.")
		com_time_end = time.time()

		# 3. 聚合更新
		if total_samples > 0:
			sum_grad = torch.stack(weighted_grads).sum(dim=0)
			avg_grad = sum_grad / total_samples

			self.update_module(self.module, self.optimizer, self.lr, avg_grad)
		else:
			print("Warning: No gradients to update (Total samples = 0).")

		cal_time_end = time.time()
		self.communication_time += com_time_end - com_time_start
		self.computation_time += cal_time_end - cal_time_start

	def run(self):
		# 设置遗忘客户端的 Loss 为 Unlearning Cross Entropy (UCE)
		for client in self.client_list:
			if client.unlearn_flag:
				client.criterion = self.UnLearningCELoss()

		# === Phase 1: 子空间预热 (Warm-up) ===
		if self.U is None:
			self.init_subspace()

		print(f"=== Start Warm-up Phase for {self.warmup_rounds} rounds ===")
		retained_clients = [c for c in self.client_list if not c.unlearn_flag]

		for j in range(self.warmup_rounds):
			if len(retained_clients) > 0:
				num_participate = int(len(retained_clients) * self.sampling_rate)
				num_participate = max(1, num_participate)
				num_participate = min(len(retained_clients), num_participate)

				choose_indices = np.random.choice(len(retained_clients), num_participate, replace=False)
				current_clients = [retained_clients[i] for i in choose_indices]

				# 打印 ID
				selected_ids = [c.id for c in current_clients]
				print(f"Warm-up Round {j + 1}/{self.warmup_rounds}: Selected Client IDs: {selected_ids}")

				_, _, g_locals = self.train(target_client_list=current_clients)

				# 【修改3】：将参与 Warm-up 的 current_clients 传给 update_subspace
				self.update_subspace(g_locals, current_clients)
			print(f"Warm-up Round {j + 1}/{self.warmup_rounds} Done.")

		print("=== Warm-up Phase Finished. Subspace Ready. ===")

		# === Phase 2: 遗忘更新 (Unlearning) ===
		print("=== Start Unlearning Phase ===")

		while not self.terminated():
			self.train_a_round()
