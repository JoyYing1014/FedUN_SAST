import sast as fs
from sast.lightsecagg.FedUN_SecAgg_Client import FedUN_SecAgg_Client
from sast.lightsecagg.FedUN_SecAgg_Server import FedUN_SecAgg_Server
from sast.lightsecagg.FedAvg_SecAgg_Server import FedAvg_SecAgg_Server

fs.Client = FedUN_SecAgg_Client
fs.FedUN = FedUN_SecAgg_Server
fs.FedAvg = FedAvg_SecAgg_Server

if __name__ == '__main__':

    my_task = fs.Fedunlearning()
    # ================== 拦截验证测试 ==================
    print("\n" + "=" * 50)
    print("【拦截验证】当前 fs.FedAvg 指向的类是:", fs.FedAvg)  # 新增这一行
    print("【拦截验证】当前使用的算法类是:", type(my_task.algorithm))
    print("【拦截验证】当前使用的客户端类是:", type(my_task.algorithm.client_list[0]))
    print("=" * 50 + "\n")
    # =================================================
    my_task.run()
