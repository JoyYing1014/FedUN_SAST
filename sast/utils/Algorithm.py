import sast as fs
import os
import numpy as np
import random
import torch
import copy
import json
import traceback
from torch.utils.tensorboard import SummaryWriter
import datetime
import csv

class Algorithm:

    def __init__(self,
                 name='Algorithm',
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
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if client_list is not None:
            client_num = len(client_list)
        elif client_list is None and client_num is not None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [fs.Client(i, module, device, train_setting) for i in range(client_num)]  
            data_loader.allocate(client_list)  
        elif client_num is None and client_list is None:
            raise RuntimeError('Both of client_num and client_list cannot be None or not None.')
        if online_client_num is None:
            online_client_num = client_num
        
        if dishonest is not None:
            dishonest_indices = np.random.choice(client_num, dishonest['dishonest_num'] ,replace=False).tolist()
            for idx, client in enumerate(client_list):
                if idx in dishonest_indices:
                    client.dishonest = dishonest
        
        choose_client_indices = np.random.choice(client_num, online_client_num, replace=False).tolist()
        self.online_client_list = [client_list[i] for i in choose_client_indices]
        if client_num > online_client_num:
            print(choose_client_indices)
        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' g_clip' + str(train_setting['g_clip'])
        if max_comm_round is None:
            max_comm_round = 10**10
        if max_training_num is None:
            max_training_num = 10**10
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.module = module
        self.model_params = None  
        self.train_setting = train_setting
        self.client_num = client_num
        self.client_list = client_list
        self.online_client_num = online_client_num
        self.save_model = save_model
        self.max_comm_round = max_comm_round
        self.max_training_num = max_training_num
        self.epochs = epochs
        self.save_name = save_name
        self.outFunc = outFunc
        self.current_comm_round = 0
        self.current_training_num = 0
        self.module.model.to(device)
        self.old_module = copy.deepcopy(self.module)  
        self.write_log = write_log
        self.params = params
        self.dishonest = dishonest
        self.test_conflicts = test_conflicts
        self.save_folder=''
        
        self.out_log = ""
        self.algorithm_log = {'descent_log': [],  
                              'layer_descent_log': [],  
                             }
        self.metric_log =  {'client_metric_history': [],
                           }
        
        self.lr = self.train_setting['optimizer'].defaults['lr']
        self.initial_lr = self.lr
        
        self.optimizer = train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, self.module.model.parameters()), lr=self.lr)
        self.optimizer.defaults = train_setting['optimizer'].defaults
        
        self.result_module = None
        
        self.test_interval = 1
        
        self.communication_time = 0
        self.computation_time = 0
        
        if params is None:
            self.test_module = 'module'
        else:
            self.test_module = params['test_module']
        
        self.pretrained_model_folder = None  
        self.model_save_name = None

        # 新增：生成基于运行日期和时间的唯一标识符 (例如：20231027_153022)
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_writer = None

    def test_global(self, model, test_loader):
        """
        在全局测试集上评估模型 (带有智能维度适配功能)
        """
        model.eval()
        correct = 0
        total = 0
        device = self.device

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # --- 核心修复：智能维度适配 ---
                try:
                    # 尝试直接前向传播 (适用于 CNN、ResNet 等需要 4D 图像的模型)
                    outputs = model(data)
                except RuntimeError as e:
                    # 如果报错提示矩阵形状不匹配，说明是 MLP 遇到了未展平的图片
                    if "mat1 and mat2" in str(e) or "size mismatch" in str(e) or "shape" in str(e).lower():
                        # 动态展平数据：把 [Batch, Channels, H, W] 压成 [Batch, Features]
                        data_flat = data.view(data.size(0), -1)
                        outputs = model(data_flat)
                    else:
                        # 如果是其他错误，不隐瞒，正常抛出
                        raise e
                # ------------------------------

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = 100.0 * correct / total
        return acc


    def reinit(self):

        self.current_comm_round = 0
        self.current_training_num = 0
        self.out_log = ""
        self.algorithm_log = {'descent_log': [],  
                              'layer_descent_log': [],  
                              }
        self.metric_log = {'client_metric_history': [],
                           }
        self.lr = self.train_setting['optimizer'].defaults['lr']
        self.initial_lr = self.lr
        self.communication_time = 0
        self.computation_time = 0

    def run(self):
        raise RuntimeError('error in Algorithm: This function must be rewritten in the child class. (该函数必须在子类中被重写！)')

    def start_running(self):
        
        self.out_log += str(self.data_loader.statistic) + '\n'
        print(self.data_loader.statistic)

        # ==================== 移到这里的：初始化文件夹和 TensorBoard ====================
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        # 将 TensorBoard 日志保存在当前任务的专属文件夹下
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_folder, 'tensorboard_logs'))
        # ============================================================================
        # ==================== 新增：记录所有参数到全局实验总表 ====================
        if self.params is not None:
            # 总表我们保存在项目的根目录，方便统筹所有实验
            master_csv_path = "experiments_summary.csv"
            file_exists = os.path.isfile(master_csv_path)

            # 提取 params 字典里的所有键作为表头
            param_keys = list(self.params.keys())

            # 自定义前置列：运行ID，详细结果文件所在的绝对/相对路径
            headers = ['run_id', 'detailed_csv_path', 'save_folder'] + param_keys

            # 推断详细 CSV 文件的保存路径
            detailed_csv_path = os.path.join(self.save_folder, f"metrics_{self.run_id}.csv")

            # 组织当前行的数据
            row_data = [self.run_id, detailed_csv_path, self.save_folder] + [self.params[k] for k in param_keys]

            # 写入 CSV
            with open(master_csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)  # 如果是第一次跑，先写入表头
                writer.writerow(row_data)  # 写入本次实验的数据
        # =========================================================================
        try:
            self.run()
        except BaseException as e:  
            traceback.print_exc()
        
        if self.save_model:
            self.module.model.to("cpu")
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
            model_save_name = self.model_save_name if self.model_save_name is not None else self.save_folder + self.save_name + '.pth'
            torch.save(self.module.model.state_dict(), model_save_name)
            self.module.model.to(self.device)  

    @staticmethod
    def update_learning_rate(optimizer, lr):

        optimizer.defaults['lr'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate(self):
        self.lr = self.initial_lr * self.train_setting['lr_decay']**self.current_comm_round
        self.update_learning_rate(self.optimizer, self.lr)  

    def update_module(self, module, optimizer, lr, g):

        self.update_learning_rate(optimizer, lr)
        for i, p in enumerate(module.model.parameters()):
            p.grad = g[module.Loc_reshape_list[i]]  
        optimizer.step()

    @staticmethod
    def terminate_extra_execute(algorithm):

        algorithm.model_params = algorithm.module.span_model_params_to_vec()  

    def terminated(self):

        self.terminate_extra_execute(self)  
        
        self.adjust_learning_rate()
        
        if self.current_comm_round % self.test_interval == 0:
            self.test()  
            
            if callable(self.outFunc):
                self.outFunc(self)
        
        self.order_free_memory()
        if self.current_comm_round >= self.max_comm_round or self.current_training_num >= self.max_training_num:
            if self.current_comm_round % self.test_interval != 0:
                
                self.test()
                if callable(self.outFunc):
                    self.outFunc(self)
            return True
        else:
            
            if self.online_client_num < self.client_num:
                choose_client_indices = np.random.choice(self.client_num, self.online_client_num, replace=False).tolist()
                print(f'Online Client: {choose_client_indices}')
                self.online_client_list = [self.client_list[i] for i in choose_client_indices]
            self.current_comm_round += 1  
            self.old_module.clone_from(self.module)  
            return False

    def weight_aggregate(self, m_locals, weights=None, update_module=True, target_client_list=None):

        if weights is None:
            
            weights = torch.Tensor(self.get_clinet_attr('local_training_number', target_client_list)).float().to(self.device)
            
            weights = weights / torch.sum(weights)
        
        params_mat = torch.stack([m_local.span_model_params_to_vec() for m_local in m_locals])  
        aggregate_params = weights @ params_mat  
        if update_module:
            self.module.reshape_vec_to_model_params(aggregate_params)  
        return aggregate_params

    def order_free_memory(self, target_client_list=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        for client in target_client_list:
            msg = {'command': 'free_memory'}
            client.get_message(msg)

    def get_loss(self, target_client_list=None, target_module=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        if target_module is None:
            target_module = self.module
        l_locals = []  
        for idx, client in enumerate(target_client_list):
            
            msg = {'command': 'cal_loss', 'target_module': target_module}
            client.get_message(msg)
            
            msg = {'command': 'require_loss'}

            msg = client.get_message(msg)
            l_locals.append(msg['l_local'])
        
        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return l_locals

    def evaluate(self, target_client_list=None, target_module=None):
        if target_client_list is None:
            target_client_list = self.online_client_list
        if target_module is None:
            target_module = self.module
        g_locals = []  
        l_locals = []  
        for idx, client in enumerate(target_client_list):
            
            msg = {'command': 'cal_gradient_loss', 'epochs': self.epochs, 'lr': self.lr, 'target_module': target_module}
            client.get_message(msg)
            
            msg = {'command': 'require_gradient_loss'}
            msg = client.get_message(msg)
            g_locals.append(msg['g_local'])
            l_locals.append(msg['l_local'])
        
        g_locals = torch.stack([g_locals[i] for i in range(len(g_locals))])
        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return g_locals, l_locals

    def train(self, target_client_list=None, target_module=None):

        if target_client_list is None:
            target_client_list = self.online_client_list
        if target_module is None:
            target_module = self.module
        m_locals = []
        l_locals = []
        g_locals = []
        for idx, client in enumerate(target_client_list):
            
            msg = {'command': 'train', 'epochs': self.epochs, 'lr': self.lr, 'target_module': target_module}
            client.get_message(msg)
            
            msg = {'command': 'require_client_module', 'requires_grad': 'False'}
            msg = client.get_message(msg)
            m_locals.append(msg['m_local'])
            l_locals.append(msg['l_local'])
            grad = (self.model_params - msg['m_local'].span_model_params_to_vec()) / self.lr
            g_locals.append(grad)
        g_locals = torch.stack([g_locals[i] for i in range(len(g_locals))])
        l_locals = torch.Tensor(l_locals).float().to(self.device)
        return m_locals, l_locals, g_locals

    def get_clinet_attr(self, attr='local_training_number', target_client_list=None):

        if target_client_list is None:
            target_client_list = self.online_client_list
        
        attrs = []
        for idx, client in enumerate(target_client_list):
            msg = {'command': 'require_attribute_value', 'attr': attr}
            msg = client.get_message(msg)
            attrs.append(msg['attr'])
        return attrs

    def cal_vec_angle(self, vec_a, vec_b):
        return float(torch.arccos(vec_a @ vec_b / torch.norm(vec_a) / torch.norm(vec_b))) / float(np.pi) * 180

    def cal_conflicts(self, g_locals, d):
        
        descent_angles = np.zeros(self.online_client_num)
        count = 0
        client_id_list = self.get_clinet_attr('id')  
        for i in range(self.online_client_num):
            angle = self.cal_vec_angle(g_locals[i], d)
            if angle > 90:
                print(f'conflict client: {client_id_list[i]}')
                descent_angles[i] = (angle - 90) / 90 * 100  
                count += 1
        if count > 0:
            self.algorithm_log['descent_log'].append((count, round(float(np.sum(descent_angles) / count), 2)))
        else:
            self.algorithm_log['descent_log'].append((count, 0))
        
        layer_descent_dict = {}
        for l, layer_indices in enumerate(self.module.Loc_list):
            d_layer = d[layer_indices]
            descent_angles = np.zeros(self.online_client_num)
            count = 0
            for i in range(self.online_client_num):
                angle = self.cal_vec_angle(g_locals[i][layer_indices], d_layer)
                if angle > 90:
                    descent_angles[i] = (angle - 90) / 90 * 100  
                    count += 1
            if count > 0:
                layer_descent_dict[str(l)] = (count, round(float(np.sum(descent_angles) / count), 2))
            else:
                layer_descent_dict[str(l)] = (count, 0)
        self.algorithm_log['layer_descent_log'].append(layer_descent_dict)

    def test(self):
        
        self.metric_log['client_metric_history'] = []
        for idx, client in enumerate(self.client_list):
            
            msg = {'command': 'test'}
            if self.test_module == 'module':  
                msg['test_module'] = self.module
            elif self.test_module == 'result_module':
                msg['test_module'] = self.result_module
            elif self.test_module == 'upload_module':
                msg['test_module'] = 'upload_module'
            client.get_message(msg)
            
            msg = {'command': 'require_test_result'}
            msg = client.get_message(msg)
            self.metric_log['client_metric_history'].append(msg['metric_history'])
        
        if self.write_log:
            self.save_log()

    def save_log(self, save_name=None):
        if save_name is None:
            save_name = self.save_name
        
        save_dict = {'params': self.params}
        save_dict['passed_communication_round'] = self.current_comm_round
        save_dict['current_online_client_ids'] = str([online_client.id for online_client in self.online_client_list])
        save_dict['algorithm_log'] = self.algorithm_log
        save_dict['metric_log'] = self.metric_log
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        file_name = self.save_folder + save_name + '.json'
        fileObject = open(file_name, 'w', encoding='utf-8')
        fileObject.write(json.dumps(save_dict))
        fileObject.close()
        file_name = self.save_folder + 'log_' + save_name + '.log'
        fileObject = open(file_name, 'w', encoding='utf-8')
        fileObject.write(self.out_log)
        fileObject.close()
