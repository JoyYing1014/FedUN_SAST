import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class MIAEvaluator:
	def __init__(self, device):
		self.device = device

	def _get_posteriors(self, model, dataloader):
		"""
		获取模型在给定数据集上的预测概率分布
		"""
		model.eval()
		posteriors = []
		with torch.no_grad():
			for data, target in dataloader:
				# 兼容不同的数据格式
				if isinstance(data, torch.Tensor):
					data = data.to(self.device)

				# 前向传播
				outputs = model(data)

				# 计算 Softmax 概率
				prob = F.softmax(outputs, dim=1)

				# 【核心技巧】：对概率进行降序排序。
				# 这样可以消除“具体是哪个类别”的干扰，影子分类器只关注“最大概率有多大、第二大概率有多大”
				prob, _ = torch.sort(prob, descending=True)
				posteriors.append(prob.cpu().numpy())

		if len(posteriors) == 0:
			return np.array([])
		return np.concatenate(posteriors, axis=0)

	def calc_mia_accuracy(self, model, member_dataloader, non_member_dataloader):
		"""
		计算 MIA 攻击准确率
		:param model: 待评估的全局模型
		:param member_dataloader: 遗忘客户端的训练数据 (成员，Label=1)
		:param non_member_dataloader: 其他干净数据的测试集 (非成员，Label=0)
		:return: MIA 准确率百分比 (完美遗忘应逼近 50%)
		"""
		# 1. 提取特征（预测概率）
		member_probs = self._get_posteriors(model, member_dataloader)
		non_member_probs = self._get_posteriors(model, non_member_dataloader)

		if len(member_probs) == 0 or len(non_member_probs) == 0:
			return 0.0

		# 2. 打标签：训练集数据为 1，测试集数据为 0
		X = np.concatenate([member_probs, non_member_probs], axis=0)
		y = np.concatenate([np.ones(len(member_probs)), np.zeros(len(non_member_probs))])

		# 3. 严格平衡正负样本数量 (非常重要，否则 50% 基准线就失去意义了)
		min_len = min(len(member_probs), len(non_member_probs))
		idx_members = np.random.choice(len(member_probs), min_len, replace=False)
		idx_non_members = np.random.choice(len(non_member_probs), min_len, replace=False) + len(member_probs)

		balanced_idx = np.concatenate([idx_members, idx_non_members])
		X_balanced = X[balanced_idx]
		y_balanced = y[balanced_idx]

		# 如果样本量太小，直接返回随机猜的水平
		if len(X_balanced) < 10:
			return 50.0

		# 4. 划分攻击模型的训练集和测试集 (70% 用于训练影子分类器，30% 用于测试攻击效果)
		X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

		# 5. 训练逻辑回归作为 MIA 攻击器
		# 适当调大 max_iter 防止收敛警告
		attacker = LogisticRegression(max_iter=2000, solver='lbfgs')
		attacker.fit(X_train, y_train)

		# 6. 计算攻击准确率并返回百分比
		acc = attacker.score(X_test, y_test)
		return acc * 100.0