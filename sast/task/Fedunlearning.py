import sast as fs
import numpy as np
import torch
import os
import csv
import copy
from torchvision.transforms import transforms
transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class Fedunlearning(fs.BasicTask):

    def __init__(self, name='Fedunlearning'):
        super().__init__(name)
        
        self.algorithm.save_folder = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/UN' + str(self.params['unlearn_cn']) + '/E' + str(self.params['E']) + '/C' + str(self.params['C']) + '/' + self.params['algorithm'] + '/'
        unlearn_pretrain_flag = self.params['unlearn_pretrain']

        # 直接定义包含参数 C 的新路径
        pretrained_model_folder = self.name + '/' + self.params[
            'module'] + '/' + self.data_loader.nickname + '/UN' + str(self.params['unlearn_cn']) + '/E' + str(
            self.params['E']) + '/C' + str(self.params['C']) + '/'
        self.algorithm.pretrained_model_folder = pretrained_model_folder
        # 定义模型完整路径
        model_path = pretrained_model_folder + f'seed{self.params["seed"]}_unlearn_task_pretrained_model.pth'
        if not unlearn_pretrain_flag:
            # ================= [执行遗忘：仅从新路径读取] =================
            if not os.path.isfile(model_path):
                # 如果新路径下没有模型，直接报错，不再寻找旧模型
                raise RuntimeError(f'未在路径下找到预训练模型: {model_path}。请先执行预训练。')
            print(f"[Info] 正在加载模型: {model_path}")
            self.algorithm.module.model.load_state_dict(torch.load(model_path))
            self.algorithm.init_model_params = self.algorithm.module.span_model_params_to_vec()
            self.algorithm.model_params = self.algorithm.module.span_model_params_to_vec()
        else:
            # ================= [执行预训练：保存至新路径] =================
            self.algorithm.save_model = True
            # 自动创建包含 C 的目录层级
            if not os.path.exists(pretrained_model_folder):
                os.makedirs(pretrained_model_folder)
            self.algorithm.model_save_name = model_path
            if isinstance(self.algorithm, fs.UnlearnAlgorithm):
                raise RuntimeError(f'当 unlearn_pretrain_flag=True 时，不能运行遗忘类算法。')
            self.algorithm.terminate_extra_execute = self.terminate_extra_execute


        # pretrained_model_folder = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/UN' + str(self.params['unlearn_cn']) + '/E' + str(self.params['E']) + '/'
        # self.algorithm.pretrained_model_folder = pretrained_model_folder
        # if not unlearn_pretrain_flag:
        #     model_path = pretrained_model_folder + f'seed{self.params["seed"]}_unlearn_task_pretrained_model.pth'
        #     if not os.path.exists(pretrained_model_folder):
        #         os.makedirs(pretrained_model_folder)
        #     if not os.path.isfile(model_path):
        #         raise RuntimeError(f'Please put the pretrained model in the path {model_path}.')
        #     self.algorithm.module.model.load_state_dict(torch.load(model_path))
        #     self.algorithm.init_model_params = self.algorithm.module.span_model_params_to_vec()
        #     self.algorithm.model_params = self.algorithm.module.span_model_params_to_vec()
        # else:
        #     self.algorithm.save_model = True
        #     self.algorithm.model_save_name = pretrained_model_folder + f'seed{self.params["seed"]}_unlearn_task_pretrained_model.pth'
        #
        #     if isinstance(self.algorithm, fs.UnlearnAlgorithm):
        #         raise RuntimeError(f'When setting unlearn_pretrain_flag=True, you cannot run unlearning FL algorithm.')
        #
        #     self.algorithm.terminate_extra_execute = self.terminate_extra_execute

        # 将字符串转为布尔值
        enable_backdoor = eval(self.params['enable_backdoor'])
        self.params['unlearn_client_id_list'] = np.random.choice(self.algorithm.client_num, self.params['unlearn_cn'], replace=False).tolist()
        print('Unlearn clients:', self.params['unlearn_client_id_list'])
        self.algorithm.out_log += f"Unlearn clients: {self.params['unlearn_client_id_list']}"


        for client in self.algorithm.client_list:
            
            if client.id in self.params['unlearn_client_id_list']:
                setattr(client, "unlearn_flag", True)
                # 只有在 enable_backdoor 为 True 时才注入后门
                if enable_backdoor:
                    pretrain_attack_portion = 1.0
                    # pretrain_attack_portion = 0.8 if unlearn_pretrain_flag else 1.0
                    self.modify_client(client, pretrain_attack_portion)
                else:
                    # 关闭后门：将后门数据置为 None，防止测试模块计算伪 ASR
                    print(f"Client {client.id} is unlearning target but BACKDOOR IS DISABLED.")
                    setattr(client, "local_backdoor_test_data", None)
                    setattr(client, "local_backdoor_test_number", 0)
                    setattr(client, "backdoor_setting", None)
                    # 确保基础测试集也被设置 如果不开启后门，仅将训练数据同步给测试数据，确保评估正常
                    client.local_test_data = copy.deepcopy(client.local_training_data)
                    client.local_test_number = client.local_training_number
                    # 仍需设置 test 对象以防报错，但使用基础的 ClientTest
                    client.test = self.ClientTest(self.algorithm.train_setting, self.algorithm.device)
            else:
                setattr(client, "unlearn_flag", False)

    @staticmethod
    def terminate_extra_execute(alg):
        alg.__class__.__bases__[0].terminate_extra_execute(alg)  

    def modify_client(self, client, attack_portion):

        # 【修改点 1】：在下毒之前，先把纯净的训练集备份下来，作为干净测试集
        clean_train_data_backup = copy.deepcopy(client.local_training_data)

        # 拷贝一份准备做后门测试
        setattr(client, "local_backdoor_test_data", copy.deepcopy(client.local_training_data))
        setattr(client, "local_backdoor_test_number", client.local_training_number)

        backdoor = fs.FigRandBackdoor(dataloader=self.algorithm.data_loader,
                                      save_folder=self.algorithm.pretrained_model_folder + 'backdoors/',
                                      save_name=f'client_{client.id}_backdoor')

        # 开始下毒
        backdoor.add_backdoor(client.local_training_data, attack_portion=attack_portion)
        backdoor.add_backdoor(client.local_backdoor_test_data, attack_portion=attack_portion)
        setattr(client, "backdoor_setting", backdoor)

        # 【修改点 2】：把刚才备份的纯净数据，赋给 local_test_data
        client.local_test_data = clean_train_data_backup
        client.local_test_number = client.local_training_number

        client.test = self.ClientTest(self.algorithm.train_setting, self.algorithm.device)

    @staticmethod
    class ClientTest:
        def __init__(self, train_setting, device):
            self.train_setting = train_setting
            self.device = device
            
            self.metric_history = {'training_loss': [], 'test_loss': [], 'local_test_number': 0, 'test_accuracy': [], 'backdoor_test_loss': [], 'backdoor_test_accuracy': []}

        def run(self, client):
            client.test_module.model.eval()
            criterion = self.train_setting['criterion'].to(self.device)
            
            self.metric_history['training_loss'].append(float(client.upload_loss) if client.upload_loss is not None else None)
            
            metric_dict = {'test_loss': 0, 'correct': 0}
            
            correct_metric = fs.Correct()
            
            with torch.no_grad():
                
                self.metric_history['local_test_number'] = client.local_test_number
                for [batch_x, batch_y] in client.local_test_data:
                    batch_x = fs.Module.change_data_device(batch_x, self.device)
                    batch_y = fs.Module.change_data_device(batch_y, self.device)
                    
                    out = client.test_module.model(batch_x)  
                    loss = criterion(out, batch_y).item()
                    metric_dict['test_loss'] += float(loss) * batch_y.shape[0]
                    # # 1. 不要在这里取 .item()
                    # loss = criterion(out, batch_y)
                    # # 2. 使用 .detach() 取出数值张量在 GPU 上直接累加
                    # metric_dict['test_loss'] += (loss.detach() * batch_y.shape[0])
                    metric_dict['correct'] += correct_metric.calc(out, batch_y)
                
                self.metric_history['test_loss'].append(round(metric_dict['test_loss'] / client.local_test_number, 4))
                # # 循环外最终保存时，再 .item() 转为普通的 Python 浮点数
                # self.metric_history['test_loss'].append(
                #     round(metric_dict['test_loss'].item() / client.local_test_number, 4))
                self.metric_history['test_accuracy'].append(100 * metric_dict['correct'] / client.local_test_number)
                # 针对后门测试数据的安全处理
                if hasattr(client, 'local_backdoor_test_data') and client.local_backdoor_test_data is not None:
                    metric_dict = {'test_loss': 0, 'correct': 0}
                    for [batch_x, batch_y] in client.local_backdoor_test_data:
                        batch_x = fs.Module.change_data_device(batch_x, self.device)
                        batch_y = fs.Module.change_data_device(batch_y, self.device)
                        out = client.test_module.model(batch_x)
                        loss = criterion(out, batch_y).item()
                        metric_dict['test_loss'] += float(loss) * batch_y.shape[0]
                        # # 1. 不要在这里取 .item()
                        # loss = criterion(out, batch_y)
                        # # 2. 使用 .detach() 取出数值张量在 GPU 上直接累加
                        # metric_dict['test_loss'] += (loss.detach() * batch_y.shape[0])
                        metric_dict['correct'] += correct_metric.calc(out, batch_y)

                    self.metric_history['backdoor_test_loss'].append(round(metric_dict['test_loss'] / client.local_backdoor_test_number, 4))
                    # self.metric_history['backdoor_test_loss'].append(round(metric_dict['test_loss'].item() / client.local_backdoor_test_number, 4))
                    self.metric_history['backdoor_test_accuracy'].append(100 * metric_dict['correct'] / client.local_backdoor_test_number)
                else:
                    # 如果没有后门数据，填充 None 而不是 0.0，避免污染指标
                    self.metric_history['backdoor_test_loss'].append(None)
                    self.metric_history['backdoor_test_accuracy'].append(None)
    # @staticmethod
    # def outFunc(alg):
    #     unlearned_client_loss_list = []
    #     retained_client_loss_list = []
    #     for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
    #         training_loss = metric_history['training_loss'][-1]
    #         if training_loss is None:
    #             continue
    #         if alg.client_list[i].unlearn_flag:
    #             unlearned_client_loss_list.append(training_loss)
    #         else:
    #             retained_client_loss_list.append(training_loss)
    #
    #     unlearned_client_local_acc_list = []
    #     retained_client_local_acc_list = []
    #     for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
    #         test_acc = metric_history['test_accuracy'][-1]
    #         if alg.client_list[i].unlearn_flag:
    #             unlearned_client_local_acc_list.append(test_acc)
    #         else:
    #             retained_client_local_acc_list.append(test_acc)
    #
    #     unlearned_client_local_backdoor_acc_list = []
    #     for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
    #         if alg.client_list[i].unlearn_flag:
    #             test_acc = metric_history['backdoor_test_accuracy'][-1]
    #             unlearned_client_local_backdoor_acc_list.append(test_acc)
    #     unlearned_client_local_acc_list = np.array(unlearned_client_local_acc_list)
    #     retained_client_local_acc_list = np.array(retained_client_local_acc_list)
    #     unlearned_client_local_backdoor_acc_list = np.array(unlearned_client_local_backdoor_acc_list)
    #
    #     def cal_fairness(values):
    #         p = np.ones(len(values))
    #         fairness = np.arccos(values @ p / (np.linalg.norm(values) * np.linalg.norm(p)))
    #         return fairness
    #     unlearned_client_fairness = cal_fairness(unlearned_client_local_acc_list)
    #     retained_client_fairness = cal_fairness(retained_client_local_acc_list)
    #
    #
    #     out_log = ""
    #     out_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
    #     out_log += 'Lr: ' + str(alg.lr) + '\n'
    #     out_log += 'round {}'.format(alg.current_comm_round) + ' training_num {}'.format(alg.current_training_num) + '\n'
    #     out_log += f'Unlearned Client Mean Global Test loss: {format(np.mean(unlearned_client_loss_list), ".6f")}' + '\n' if len(unlearned_client_loss_list) > 0 else ''
    #     out_log += f'Unlearned Client Local Test Acc: {format(np.mean(unlearned_client_local_acc_list/100), ".3f")}({format(np.std(unlearned_client_local_acc_list/100), ".3f")}), angle: {format(unlearned_client_fairness, ".6f")}, min: {format(np.min(unlearned_client_local_acc_list), ".6f")}, max: {format(np.max(unlearned_client_local_acc_list), ".6f")}' + '\n' if len(unlearned_client_local_acc_list) > 0 else ''
    #     out_log += f'ASR: {format(np.mean(unlearned_client_local_backdoor_acc_list/100), ".3f")}({format(np.std(unlearned_client_local_backdoor_acc_list/100), ".3f")}), min: {format(np.min(unlearned_client_local_backdoor_acc_list), ".6f")}, max: {format(np.max(unlearned_client_local_backdoor_acc_list), ".6f")}' + '\n' if len(unlearned_client_local_backdoor_acc_list) > 0 else ''
    #     out_log += f'Retained Client Mean Global Test loss: {format(np.mean(retained_client_loss_list), ".6f")}' + '\n' if len(retained_client_loss_list) > 0 else ''
    #     out_log += f'Retained Client Local Test Acc: {format(np.mean(retained_client_local_acc_list/100), ".3f")}({format(np.std(retained_client_local_acc_list/100), ".3f")}), angle: {format(retained_client_fairness, ".6f")}, min: {format(np.min(retained_client_local_acc_list), ".6f")}, max: {format(np.max(retained_client_local_acc_list), ".6f")}' + '\n'
    #     out_log += f'communication_time: {alg.communication_time}, computation_time: {alg.computation_time} \n'
    #     out_log += '\n'
    #     # --- 新增的 CSV 记录逻辑开始 ---
    #     csv_file_name = f"metrics_{alg.run_id}.csv"
    #     csv_file_path = os.path.join(alg.save_folder, csv_file_name)
    #     file_exists = os.path.isfile(csv_file_path)
    #     # 提取关键指标用于写入 CSV
    #     round_num = alg.current_comm_round
    #     # 遗忘客户端指标 (防止为空)
    #     unl_loss = np.mean(unlearned_client_loss_list) if len(unlearned_client_loss_list) > 0 else 0
    #     unl_acc = np.mean(unlearned_client_local_acc_list) if len(unlearned_client_local_acc_list) > 0 else 0
    #     asr = np.mean(unlearned_client_local_backdoor_acc_list) if len(
    #         unlearned_client_local_backdoor_acc_list) > 0 else 0
    #     # 保留客户端指标
    #     ret_loss = np.mean(retained_client_loss_list) if len(retained_client_loss_list) > 0 else 0
    #     ret_acc = np.mean(retained_client_local_acc_list) if len(retained_client_local_acc_list) > 0 else 0
    #     with open(csv_file_path, mode='a', newline='') as f:
    #         writer = csv.writer(f)
    #         if not file_exists:
    #             writer.writerow(
    #                 ['Round', 'Unlearned_Loss', 'Unlearned_Acc(%)', 'ASR(%)', 'Retained_Loss', 'Retained_Acc(%)',
    #                  'Comm_Time', 'Comp_Time'])
    #         writer.writerow(
    #             [round_num, unl_loss, unl_acc, asr, ret_loss, ret_acc, alg.communication_time, alg.computation_time])
    #     # --- 新增的 CSV 记录逻辑结束 ---
    #     alg.out_log = out_log + alg.out_log
    #     print(str(alg.name))
    #     print(out_log)

    # ==========================================
    # 1. 原有的本地指标统计逻辑 (Loss, Acc, ASR)
    # ==========================================

    @staticmethod
    def outFunc(alg):
        # 读取是否启用了后门
        enable_backdoor = str(alg.params.get('enable_backdoor', 'True')).lower() == 'true'
        unlearned_client_loss_list = []
        retained_client_loss_list = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            training_loss = metric_history['training_loss'][-1]
            if training_loss is None:
                continue
            if alg.client_list[i].unlearn_flag:
                unlearned_client_loss_list.append(training_loss)
            else:
                retained_client_loss_list.append(training_loss)

        unlearned_client_local_acc_list = []
        retained_client_local_acc_list = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            test_acc = metric_history['test_accuracy'][-1]
            if alg.client_list[i].unlearn_flag:
                unlearned_client_local_acc_list.append(test_acc)
            else:
                retained_client_local_acc_list.append(test_acc)

        unlearned_client_local_backdoor_acc_list = []
        # 仅在启用后门时提取 backdoor accuracy
        if enable_backdoor:
            for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
                if alg.client_list[i].unlearn_flag:
                    test_acc = metric_history['backdoor_test_accuracy'][-1]
                    if test_acc is not None:
                        unlearned_client_local_backdoor_acc_list.append(test_acc)

        unlearned_client_local_acc_list = np.array(unlearned_client_local_acc_list)
        retained_client_local_acc_list = np.array(retained_client_local_acc_list)
        unlearned_client_local_backdoor_acc_list = np.array(unlearned_client_local_backdoor_acc_list)

        def cal_fairness(values):
            p = np.ones(len(values))
            fairness = np.arccos(values @ p / (np.linalg.norm(values) * np.linalg.norm(p)))
            return fairness

        unlearned_client_fairness = cal_fairness(unlearned_client_local_acc_list) if len(
            unlearned_client_local_acc_list) > 0 else 0
        retained_client_fairness = cal_fairness(retained_client_local_acc_list) if len(
            retained_client_local_acc_list) > 0 else 0

        # ==========================================
        # 2. 全局测试准确率 (Global Test Acc)
        # ==========================================
        global_acc = 0.0
        try:
            global_test_loader = alg.data_loader.get_global_test_data()
            global_acc = alg.test_global(alg.module.model, global_test_loader)
        except AttributeError:
            print("⚠️ 警告: 未在 DataLoader 中找到 get_global_test_data 或 Algorithm 中找到 test_global。跳过全局测试。")

        # ==========================================
        # 3. 智能触发 MIA 评估
        # ==========================================
        mia_acc = None
        pretrain_val = str(alg.params.get('unlearn_pretrain', 'False')).lower()
        is_pretraining = pretrain_val == 'true'
        # do_mia_test = False
        do_mia_test = True

        # # 智能触发判定
        # if not is_pretraining:
        #     # 在遗忘阶段 (SAST/FedOSD)，每次测试间隔都测 MIA，以记录遗忘轨迹
        #     do_mia_test = True
        # else:
        #     # 在预训练阶段，只在最后 5 轮测 MIA 以节省时间，确立 Baseline
        #     if alg.current_comm_round >= alg.max_comm_round - 5:
        #         do_mia_test = True

        if do_mia_test:
            mia_evaluator = fs.MIAEvaluator(alg.device)
            # 找一个保留客户端的测试集作为 Non-member
            retained_test_data = None
            for client in alg.client_list:
                if not getattr(client, "unlearn_flag", False):
                    retained_test_data = client.local_test_data
                    break

            mia_acc_list = []
            if retained_test_data is not None:
                for client in alg.client_list:
                    if getattr(client, "unlearn_flag", False):
                        # 遗忘客户端的本地训练集作为 Member
                        member_data = client.local_training_data
                        acc = mia_evaluator.calc_mia_accuracy(alg.module.model, member_data, retained_test_data)
                        mia_acc_list.append(acc)

            if len(mia_acc_list) > 0:
                mia_acc = np.mean(mia_acc_list)

        # ==========================================
        # 4. 时间累加器
        # ==========================================
        # 在 alg 对象上动态挂载总时间属性，累加每一轮的时间
        if not hasattr(alg, 'total_comm_time_accumulated'):
            alg.total_comm_time_accumulated = 0.0
            alg.total_comp_time_accumulated = 0.0

        alg.total_comm_time_accumulated += alg.communication_time
        alg.total_comp_time_accumulated += alg.computation_time

        # ==========================================
        # 5. 控制台日志格式化输出
        # ==========================================
        out_log = ""
        out_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
        out_log += 'Lr: ' + str(alg.lr) + '\n'
        out_log += 'round {}'.format(alg.current_comm_round) + ' training_num {}'.format(alg.current_training_num) + '\n'

        # 打印新增的 Global Acc 和 MIA
        out_log += f'🌐 [Global] 10-Class Test Acc: {global_acc:.2f}%\n'
        if mia_acc is not None:
            out_log += f'🔥 [Privacy] MIA Accuracy: {mia_acc:.2f}%\n'

        out_log += f'Unlearned Client Mean Global Test loss: {format(np.mean(unlearned_client_loss_list), ".6f")}' + '\n' if len(
            unlearned_client_loss_list) > 0 else ''
        out_log += f'Unlearned Client Local Test Acc: {format(np.mean(unlearned_client_local_acc_list / 100), ".3f")}({format(np.std(unlearned_client_local_acc_list / 100), ".3f")}), angle: {format(unlearned_client_fairness, ".6f")}, min: {format(np.min(unlearned_client_local_acc_list), ".6f")}, max: {format(np.max(unlearned_client_local_acc_list), ".6f")}' + '\n' if len(
            unlearned_client_local_acc_list) > 0 else ''
        # 仅当开启后门时才输出 ASR 日志
        if enable_backdoor and len(unlearned_client_local_backdoor_acc_list) > 0:
            out_log += f'ASR: {format(np.mean(unlearned_client_local_backdoor_acc_list / 100), ".3f")}({format(np.std(unlearned_client_local_backdoor_acc_list / 100), ".3f")}), min: {format(np.min(unlearned_client_local_backdoor_acc_list), ".6f")}, max: {format(np.max(unlearned_client_local_backdoor_acc_list), ".6f")}' + '\n' if len(
                unlearned_client_local_backdoor_acc_list) > 0 else ''
        out_log += f'Retained Client Mean Global Test loss: {format(np.mean(retained_client_loss_list), ".6f")}' + '\n' if len(
            retained_client_loss_list) > 0 else ''
        out_log += f'Retained Client Local Test Acc: {format(np.mean(retained_client_local_acc_list / 100), ".3f")}({format(np.std(retained_client_local_acc_list / 100), ".3f")}), angle: {format(retained_client_fairness, ".6f")}, min: {format(np.min(retained_client_local_acc_list), ".6f")}, max: {format(np.max(retained_client_local_acc_list), ".6f")}' + '\n'
        out_log += f'communication_time: {alg.communication_time}, computation_time: {alg.computation_time} \n\n'

        alg.out_log = out_log + alg.out_log
        print(str(alg.name))
        print(out_log)

        # ==========================================
        # 6. CSV 文件记录逻辑 (扩充字段)
        # ==========================================
        csv_file_name = f"metrics_{alg.run_id}.csv"
        csv_file_path = os.path.join(alg.save_folder, csv_file_name)
        file_exists = os.path.isfile(csv_file_path)

        round_num = alg.current_comm_round

        # 提取指标，防止空列表导致报错
        unl_loss = np.mean(unlearned_client_loss_list) if len(unlearned_client_loss_list) > 0 else 0
        unl_acc = np.mean(unlearned_client_local_acc_list) if len(unlearned_client_local_acc_list) > 0 else 0
        # asr = np.mean(unlearned_client_local_backdoor_acc_list) if len(unlearned_client_local_backdoor_acc_list) > 0 else 0
        # 没有后门时 CSV 里 ASR 记为空字符串
        asr = np.mean(unlearned_client_local_backdoor_acc_list) if (
                    enable_backdoor and len(unlearned_client_local_backdoor_acc_list) > 0) else ""
        ret_loss = np.mean(retained_client_loss_list) if len(retained_client_loss_list) > 0 else 0
        ret_acc = np.mean(retained_client_local_acc_list) if len(retained_client_local_acc_list) > 0 else 0

        # 将 None 转换为空字符串，确保没有测 MIA 的轮次在 Excel 里显示为空，而不是报错
        csv_mia = mia_acc if mia_acc is not None else ""

        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # 增加了 Global_Acc(%) 和 MIA(%) 两个表头
                writer.writerow([
                    'Round', 'Unlearned_Loss', 'Unlearned_Acc(%)', 'ASR(%)',
                    'Retained_Loss', 'Retained_Acc(%)', 'Global_Acc(%)', 'MIA(%)',
                    'Comm_Time', 'Comp_Time'
                ])
            writer.writerow([
                round_num, unl_loss, unl_acc, asr,
                ret_loss, ret_acc, global_acc, csv_mia,
                alg.communication_time, alg.computation_time
            ])

        # ==========================================
        # 7. 【新增】记录所有客户端ACC的日志 (每一轮记录)
        # 行=Round，列=Client_X_ACC
        # ==========================================
        acc_csv_file_name = f"{alg.run_id}_acc.csv"
        acc_csv_file_path = os.path.join(alg.save_folder, acc_csv_file_name)
        acc_file_exists = os.path.isfile(acc_csv_file_path)

        all_clients_acc = []
        # 按顺序遍历提取每一个客户端在该轮最新的测试准确率
        for metric_history in alg.metric_log['client_metric_history']:
            if len(metric_history['test_accuracy']) > 0:
                all_clients_acc.append(metric_history['test_accuracy'][-1])
            else:
                all_clients_acc.append(0.0)

        with open(acc_csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not acc_file_exists:
                # 第一行写入实验参数，便于追溯
                writer.writerow(['# Experiment Parameters:', str(alg.params)])
                # 写入表头：Round, Client_0_Acc, Client_1_Acc ...
                headers = ['Round'] + [f'Client_{c.id}_Acc(%)' for c in alg.client_list]
                writer.writerow(headers)

            # 写入当前轮次的各客户端 ACC
            writer.writerow([round_num] + all_clients_acc)

        # ==========================================
        # 8. 最后写入全局汇总表格的逻辑
        # ==========================================
        # 检查是否是最后一轮 (max_comm_round 是总轮数，由于轮数从 0 开始，最后一轮是 max_comm_round - 1)
        if alg.current_comm_round == alg.max_comm_round:
            # 存放在根目录或你指定的地方
            summary_csv_path = "final_results_summary.csv"
            summary_exists = os.path.isfile(summary_csv_path)

            # 判断当前是预训练还是遗忘阶段
            is_pretrain = str(alg.params.get('unlearn_pretrain', 'False')).lower() == 'true'
            stage = "Pre-train" if is_pretrain else "Unlearn"

            with open(summary_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not summary_exists:
                    # 如果文件刚创建，写入表头
                    writer.writerow([
                        'Dataset', 'N', 'NC', 'C', 'Stage',
                        'Global_Acc(%)', 'ASR(%)', 'MIA(%)',
                        'Total_Comm_Time(s)', 'Total_Comp_Time(s)', 'Retained_Acc(%)'
                    ])

                # 写入你关注的核心参数和这最后一轮的最终指标
                writer.writerow([
                    alg.params.get('dataloader', 'Unknown'),
                    alg.params.get('N', 'Unknown'),
                    alg.params.get('NC', 'Unknown'),
                    alg.params.get('C', 'Unknown'),
                    stage,
                    round(global_acc, 3),  # 保留三位小数
                    round(asr, 3) if asr != "" else "",  # 如果没有 ASR 就写入空值 后门成功率
                    round(mia_acc, 3) if mia_acc is not None else "",
                    round(alg.total_comm_time_accumulated, 2),  # 累加后的总通信时间
                    round(alg.total_comp_time_accumulated, 2),  # 累加后的总计算时间
                    # 新增的加载后面，不影响前面已经保存的
                    round(ret_acc, 3)  # 保留三位小数
                ])

    def read_params(self, return_parser=False):
        parser = super().read_params(return_parser=True)

        # 需要执行遗忘任务的客户端数量
        parser.add_argument('--unlearn_cn', help='unlearn client num', type=int, default=1)
        # 为 True 时先进行常规 FL 训练并保存模型；为 False 时加载已有的预训练模型执行遗忘
        parser.add_argument('--unlearn_pretrain', help='pretrain the model before unlearning', type=str, default=False)
        # 遗忘阶段的轮数（必须小于总轮数 R）
        parser.add_argument('--UR', help='Unlearning round, must be smaller than R', type=int, default=100)
        # 恢复阶段（Post-training）的学习率 如果设为 -1，则沿用遗忘阶段的衰减学习率
        parser.add_argument('--r_lr', help='Learning rate in the post-training', type=float, default=-1)
        # 后门控制参数
        parser.add_argument('--enable_backdoor', help='whether to enable backdoor attack', type=str, default='True')

        try:
            if return_parser:
                return parser
            else:
                params = vars(parser.parse_args())
                
                if params['UR'] > params['R']:
                    raise RuntimeError('The parameter of UR must not be bigger than R.')
                return params
        except IOError as msg:
            parser.error(str(msg))


if __name__ == '__main__':
    my_task = Fedunlearning()
    my_task.run()
