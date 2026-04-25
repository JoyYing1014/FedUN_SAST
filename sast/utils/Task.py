import sast as fs
import numpy as np
import argparse
import torch
import sys
import os

torch.multiprocessing.set_sharing_strategy('file_system')

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class BasicTask:
	
	def __init__(self, name='BasicTask'):
		self.name = name
		self.params = self.read_params()
		
		# === 【新增】配置全局日志记录 ===
		# 1. 构造包含所有参数的长文件名
		# 过滤掉一些路径相关的参数以缩短文件名，或者选择保留所有
		# 注意：文件名过长在某些系统（如Windows）可能会报错，Linux通常支持255字符
		param_str_list = []
		for k, v in self.params.items():
			# 简化一下：只保留非默认值的关键参数，或者保留全部
			# 这里按照您的要求：覆盖所有参数
			param_str_list.append(f"{k}={v}")
		
		# 用下划线连接，并截断过长的文件名以防报错
		full_param_str = ",".join(param_str_list)
		
		# 创建日志目录
		log_dir = 'logs'
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		
		# 限制文件名长度（Linux最大255字节），保留前150个字符和后50个字符，中间用hash
		if len(full_param_str) > 240:
			full_param_str = full_param_str[:150] + "_hash" + str(hash(full_param_str))[-8:]
		
		log_filename = os.path.join(log_dir, f"{full_param_str}.log")
		
		print(f"Log will be saved to: {log_filename}")
		
		# 重定向 sys.stdout
		sys.stdout = Logger(log_filename)
		# =========================================
		self.data_loader, self.algorithm = self.initialize(self.params)
		
		self.algorithm.save_folder = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/C' + str(self.params['C']) + '/' + \
		                             self.params['algorithm'] + '/'
		
		self.algorithm.save_name = 'seed' + str(self.params['seed']) + ' N' + str(self.data_loader.pool_size) + ' C' + str(
			self.params['C']) + ' ' + self.algorithm.save_name
		if self.params['load_pretrained']:
			
			model_path = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/pretrained_model.pth'
			if os.path.isfile(model_path):
				print('Find pretrained model')
				self.algorithm.module.model.load_state_dict(torch.load(model_path))
				self.algorithm.module.model.to(self.algorithm.device)
	
	def run(self):
		self.algorithm.start_running()
	
	def __str__(self):
		
		print(self.params)
	
	@staticmethod
	def outFunc(alg):
		
		loss_list = []
		for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
			training_loss = metric_history['training_loss'][-1]
			if training_loss is None:
				continue
			loss_list.append(training_loss)
		loss_list = np.array(loss_list)
		
		local_acc_list = []
		for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
			local_acc_list.append(metric_history['test_accuracy'][-1])
		local_acc_list = np.array(local_acc_list)
		p = np.ones(len(local_acc_list))
		local_acc_fairness = np.arccos(local_acc_list @ p / (np.linalg.norm(local_acc_list) * np.linalg.norm(p)))
		
		out_log = ""
		out_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
		out_log += 'Lr: ' + str(alg.lr) + '\n'
		out_log += 'round {}'.format(alg.current_comm_round) + ' training_num {}'.format(alg.current_training_num) + '\n'
		out_log += f'Mean Global Test loss: {format(np.mean(loss_list), ".6f")}' + '\n' if len(loss_list) > 0 else ''
		out_log += 'global model test: \n'
		out_log += f'Local Test Acc: {format(np.mean(local_acc_list / 100), ".3f")}({format(np.std(local_acc_list / 100), ".3f")}), angle: {format(local_acc_fairness, ".6f")}, min: {format(np.min(local_acc_list), ".6f")}, max: {format(np.max(local_acc_list), ".6f")}' + '\n'
		out_log += f'communication_time: {alg.communication_time}, computation_time: {alg.computation_time} \n'
		out_log += '\n'
		alg.out_log = out_log + alg.out_log
		print(out_log)
	
	def read_params(self, return_parser=False):
		parser = argparse.ArgumentParser()
		# 随机种子参数
		parser.add_argument('--seed', help='seed', type=int, default=1)
		# CPU/GPU数目
		parser.add_argument('--device', help='device: -1, 0, 1, or ...', type=int, default=0)
		# 模型名称
		parser.add_argument('--module', help='module name;', type=str, default='CNN_CIFAR10_FedFV')
		# 算法名称
		parser.add_argument('--algorithm', help='algorithm name;', type=str, default='FedAvg')
		# 数据集
		parser.add_argument('--dataloader', help='dataloader name;', type=str, default='DataLoader_cifar10_pat')
		# 数据标签类型的字符串标识，通常用默认的即可
		parser.add_argument('--types', help='dataloader label types;', type=str, default='default_type')
		# 批次大小
		parser.add_argument('--B', help='batch size', type=int, default=128)
		# 每个客户端的类别数
		parser.add_argument('--NC', help='client_class_num', type=int, default=2)
		# 这个参数控制的是每个客户端分到的“数据样本总数”是否相等
		parser.add_argument('--balance', help='balance or not for pathological separation', type=str, default='True')
		# 狄利克雷划分的参数。如果你的代码想使用狄利克雷分布来切分 Non-IID 数据（partition='dir'），这个值越小，数据越 Non-IID；值越大越接近 IID。
		parser.add_argument('--Diralpha', help='alpha parameter for dirichlet', type=float, default=1)
		# 总客户端数量
		parser.add_argument('--N', help='client num', type=int, default=100)
		# 每轮选择客户端的比例
		parser.add_argument('--C', help='select client proportion', type=float, default=1.0)
		# 总通信轮数
		parser.add_argument('--R', help='communication round', type=int, default=3000)
		# 本地训练轮数
		parser.add_argument('--E', help='local epochs', type=int, default=1)
		# 学习率
		parser.add_argument('--lr', help='learning rate', type=float, default=0.05)
		# 学习率衰减率 让模型在训练初期快速下降，在训练末期（接近最低点时）迈小步子，防止跨过最优解
		parser.add_argument('--decay', help='learning rate decay', type=float, default=0.995)
		# 动量 类似于“惯性”。如果连续几次梯度的方向都一致，它会加速更新；如果梯度方向来回震荡，它能平滑更新轨迹。默认设为 0.5 或 0.9 都可以帮助模型更快、更稳定地收敛。
		parser.add_argument('--momentum', help='momentum', type=float, default=0.5)
		# 测试间隔。默认 1 表示每训练 1 轮就用测试集评估一次模型。如果你觉得训练太慢，可以设为 5 或 10
		parser.add_argument('--test_interval', help='test interval', type=int, default=1)
		# 设为 True 时，代码会统计上传梯度之间的冲突（主要在 FedOSD 算法里用到，用来画论文里的冲突分析图）。
		parser.add_argument('--test_conflicts', help='test conflicts', type=str, default='False')
		# 本地更新步的类型。bgd 表示 Batch Gradient Descent（用整个本地数据集算一次梯度），sgd 表示随机梯度下降。
		parser.add_argument('--step_type', help='step type', type=str, default='bgd')
		# 测试时使用哪个模型模块（保持默认 module 即可）
		parser.add_argument('--test_module', help='test module', type=str, default='module')
		# 是否直接读取已经存在硬盘里的模型来跳过预训练（通常配合你之前看到的 --unlearn_pretrain False 使用）
		parser.add_argument('--load_pretrained', help='Load the pretrained model', type=str, default=False)
		# 梯度裁剪阈值（Gradient Clipping）。用于防止梯度爆炸。默认 -1.0 表示不裁剪。如果你的模型在遗忘时 Loss 变成 NaN，可以把这个设为 1.0 试试。
		parser.add_argument('--g_clip', help='parameter gradient clipping when using gradient ascent.', type=float, default=-1.0)
		# 是否保存模型权重
		parser.add_argument('--save_model', help='save model', type=str, default='True')
		# 是否改变数据划分规则，如NC。--recreate True 强制重新生成当前配置的 .npy 文件并覆盖旧文件
		parser.add_argument('--recreate', help='Force recreate data pool', type=str, default='False')
		# === 新增 FedUN 参数 ===
		parser.add_argument('--sast_beta', help='retained gradient weight for FedUN', type=float, default=1.0)
		parser.add_argument('--sast_gamma', help='unlearning gradient weight for FedUN', type=float, default=5.0)  # 调大默认值
		parser.add_argument('--do_projection', help='whether to do orthogonal projection', type=str, default='True')
		parser.add_argument('--k', help='dimension of the approximate subspace', type=int, default=20)
		parser.add_argument('--warmup_rounds', help='number of warm-up laps', type=int, default=10)
		parser.add_argument('--u_update_freq', help='frequency of subspace update in unlearning', type=int, default=1)

		# # 这是另一种叫 sort_and_split 的数据划分方式用到的参数，你在用 _pat 加载器时根本用不到它们。
		# parser.add_argument('--SN', help='split num', type=int, default=200)
		# parser.add_argument('--PN', help='pick num', type=int, default=2)
		# ============= 其他特定算法的超参数 =============
		# # FedMDFG 算法的公平性角度和线搜索参数
		# parser.add_argument('--theta', help='fairness angle of FedMDFG', type=float, default=11.25)
		# parser.add_argument('--s', help='line search parameter of FedMDFG', type=int, default=1)
		# # FedFV / APFL 算法的权重参数
		# parser.add_argument('--alpha', help='alpha of FedFV/APFL', type=float, default=0.1)
		# # FedFV / FedRep 算法的控制参数
		# parser.add_argument('--tau', help='parameter tau in FedFV/FedRep', type=int, default=1)
		# # FedFa 算法的超参数
		# parser.add_argument('--beta', help='beta of FedFa', type=float, default=0.5)
		# parser.add_argument('--gamma', help='parameter gamma in FedFa', type=float, default=0.9)
		# # Ditto / pFedMe / pFedGF 等个性化联邦学习算法的正则化系数
		# parser.add_argument('--lam', help='parameter tau in Ditto/FedAMP/pFedMe/pFedGF', type=float, default=0.1)
		# # FedMGDA+ 的参数
		# parser.add_argument('--epsilon', help='parameter epsilon in FedMGDA+', type=float, default=0.1)
		# # qFedAvg（公平联邦学习）的参数
		# parser.add_argument('--q', help='parameter q in qFedAvg', type=float, default=0.1)
		# # TERM 算法的参数
		# parser.add_argument('--t', help='parameter t in TERM', type=float, default=1.0)
		# # FedProx 算法的近端项正则化系数（这个比较有名，如果要用 FedProx 作为基线可以保留，否则没用）
		# parser.add_argument('--mu', help='parameter mu in FedProx', type=float, default=0.0)
		# ============= 恶意攻击与防御参数（暂时没用，除非研究系统鲁棒性） =============
		# 恶意客户端的数量
		parser.add_argument('--dishonest_num', help='dishonest number', type=int, default=0)
		# 缩放攻击（恶意放大梯度）
		parser.add_argument('--scaled_update', help='scaled update attack', type=str, default='None')
		# 随机更新攻击（客户端上传随机噪声）
		parser.add_argument('--random_update', help='random update attack', type=str, default='None')
		# 零更新攻击（客户端上传 0）
		parser.add_argument('--zero_update', help='zero update attack', type=str, default='None')
		# --use_secagg False 关闭遗忘阶段的加密
		parser.add_argument('--use_secagg', help='Whether to use SecAgg during unlearning',
		                    type=lambda x: (str(x).lower() == 'true'), default=True)

		try:
			if return_parser:
				return parser
			else:
				params = vars(parser.parse_args())
				return params
		except IOError as msg:
			parser.error(str(msg))
	
	def initialize(self, params):
		fs.setup_seed(seed=params['seed'])
		device = torch.device('cuda:' + str(params['device']) if torch.cuda.is_available() and params['device'] != -1 else "cpu")
		Module = getattr(sys.modules['sast'], params['module'])
		module = Module(device)
		Dataloader = getattr(sys.modules['sast'], params['dataloader'])
		# data_loader = Dataloader(params=params, input_require_shape=module.input_require_shape)
		data_loader = Dataloader(
			params=params,
			input_require_shape=module.input_require_shape,
			recreate=eval(params['recreate'])  # 核心修改：将命令行参数传给底层
		)
		# --- 新增：初始化全局测试集 ---
		self.global_test_loader = data_loader.get_global_test_data()
		# ---------------------------
		module.generate_model(data_loader.input_data_shape, data_loader.target_class_num)
		optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, module.model.parameters()), lr=params['lr'], momentum=params['momentum'])
		train_setting = {'criterion': torch.nn.CrossEntropyLoss(), 'optimizer': optimizer, 'lr_decay': params['decay'], 'step_type': params['step_type'],
		                 'g_clip'   : params['g_clip']}
		test_interval = params['test_interval']
		save_model = params['save_model']
		dishonest_num = params['dishonest_num']
		scaled_update = eval(params['scaled_update'])
		if scaled_update is not None:
			scaled_update = float(scaled_update)
		dishonest = {'dishonest_num': dishonest_num,
		             'scaled_update': scaled_update,
		             'random_update': eval(params['random_update']),
		             'zero_update'  : eval(params['zero_update'])}
		test_conflicts = eval(params['test_conflicts'])
		Algorithm = getattr(sys.modules['sast'], params['algorithm'])
		algorithm = Algorithm(data_loader=data_loader,
		                      module=module,
		                      device=device,
		                      train_setting=train_setting,
		                      client_num=data_loader.pool_size,
		                      online_client_num=int(data_loader.pool_size * params['C']),
		                      save_model=save_model,
		                      max_comm_round=params['R'],
		                      max_training_num=None,
		                      epochs=params['E'],
		                      outFunc=self.outFunc,
		                      write_log=True,
		                      dishonest=dishonest,
		                      test_conflicts=test_conflicts,
		                      params=params, )
		algorithm.test_interval = test_interval
		return data_loader, algorithm


if __name__ == '__main__':
	my_task = BasicTask()
	my_task.run()