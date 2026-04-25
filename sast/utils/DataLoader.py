
import copy
import random
import torch
import numpy as np


class DataLoader:

    def __init__(self,
                 name='DataLoader',
                 nickname='DataLoader',
                 pool_size=0,
                 batch_size=0,
                 input_require_shape=None,
                 *args,
                 **kwargs):
        self.name = name
        self.nickname = nickname
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.input_require_shape = input_require_shape
        self.raw_data_shape = None
        self.input_data_shape = None
        self.target_class_num = None
        self.data_pool = None
        self.global_test_pool = None

    # 新增
    def get_global_test_data(self):
        """
        实时返回完整的官方测试集，完美绕过 .npy 缓存缺失问题
        """
        import torch
        import torchvision
        import torchvision.transforms as transforms
        import sast as fs  # 导入你的核心包以获取数据路径

        b_size = getattr(self, 'batch_size', 128)

        # 基础图像转换
        transform = transforms.Compose([transforms.ToTensor()])

        name_lower = self.name.lower()

        # 根据 DataLoader 的名字，动态从硬盘/网络加载干净的全局测试集
        if 'cifar100' in name_lower:
            testset = torchvision.datasets.CIFAR100(root=fs.data_folder_path, train=False, download=True,
                                                    transform=transform)
        elif 'cifar10' in name_lower:
            testset = torchvision.datasets.CIFAR10(root=fs.data_folder_path, train=False, download=True,
                                                   transform=transform)
        elif 'fashion' in name_lower:
            testset = torchvision.datasets.FashionMNIST(root=fs.data_folder_path, train=False, download=True,
                                                        transform=transform)
        elif 'mnist' in name_lower:
            testset = torchvision.datasets.MNIST(root=fs.data_folder_path, train=False, download=True,
                                                 transform=transform)
        else:
            raise ValueError(f"无法自动识别数据集类型以加载全局测试集，当前 DataLoader 名称为: {self.name}")

        # 封装为 DataLoader 返回，不需要 shuffle
        return torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False)


    def allocate(self, client_list):

        choose_data_pool_item_indices = np.random.choice(list(range(self.pool_size)), len(client_list), replace=False)
        for idx, client in enumerate(client_list):
            data_pool_item = self.data_pool[choose_data_pool_item_indices[idx]]
            client.update_data(data_pool_item['local_training_data'],
                               data_pool_item['local_training_number'],
                               data_pool_item['local_test_data'],
                               data_pool_item['local_test_number'],
                               )

    def reshape(self, data, require_shape):

        return data.reshape(require_shape)

    def transform_data(self, dataset):

        input_data = []
        for i, data_item in enumerate(dataset):
            input_data.append(data_item[0])
        input_data = torch.cat(input_data)
        target_data = copy.deepcopy(dataset.targets)
        return input_data, target_data

    def cal_data_shape(self, raw_input_data_shape):

        self.raw_data_shape = raw_input_data_shape[1:]
        def cal(require_shape, raw_shape):
            if len(require_shape) == len(raw_shape) - 1:
                data_shape = list(raw_shape[1:])
            else:
                data_shape = []
                for i in range(1, len(raw_shape)):
                    if i < len(require_shape) + 1:
                        data_shape.append(raw_shape[i])
                    else:
                        data_shape[-1] *= raw_shape[i]
            return data_shape
        self.input_data_shape = cal(self.input_require_shape, raw_input_data_shape)

    @staticmethod
    def separate_list(input_list, n):

        def separate(input_list, n):
            for i in range(0, len(input_list), n):
                yield input_list[i: i + n]

        return list(separate(input_list, n))

    @staticmethod
    def separate_list_to_n_parts(input_list, n):

        n2 = len(input_list)
        _, choose_indices_reverse = DataLoader.random_choice(n, n2)
        results = []
        for choose_indices in choose_indices_reverse:
            result = []
            for choose_idx in choose_indices:
                result.append(input_list[choose_idx])
            results.append(result)
        return results

    @staticmethod
    def random_choice(n1, n2):

        indices = list(range(n1))
        indices_copy = copy.deepcopy(indices)
        choose_indices = []
        choose_indices_reverse = []
        for i in range(n1):
            choose_indices_reverse.append([])
        for i in range(n2):
            if len(indices_copy) == 0:
                indices_copy = copy.deepcopy(indices)
            pick = indices_copy[random.randint(0, len(indices_copy) - 1)]
            choose_indices.append(pick)
            choose_indices_reverse[pick].append(i)
            indices_copy.remove(pick)
        return choose_indices, choose_indices_reverse


if __name__ == '__main__':
    lst = [1,2,3,4,5,6,7,8,9,10]
    res_lst = DataLoader.separate_list(lst, 3)
    print(res_lst)


