# -*- coding: utf-8 -*-
import os

# import base class
from sast.utils.Algorithm import Algorithm
from sast.utils.Client import Client
from sast.utils.DataLoader import DataLoader
from sast.utils.Module import Module
from sast.utils.Metric import Metric
from sast.utils.seed import setup_seed
# import metrics
from sast.metric.Correct import Correct
from sast.metric.MAE import MAE
from sast.metric.RMSE import RMSE
from sast.metric.Precision import Precision
from sast.metric.Recall import Recall
# import models
from sast.model.LeNet5 import LeNet5
from sast.model.CNN import CNN_CIFAR10
from sast.model.MLP import MLP
from sast.model.NFResNet import NFResNet18, NFResNet50

# import algorithm
from sast.algorithm.FedAvg.FedAvg import FedAvg
# unlearning
from sast.algorithm.unlearning.UnlearnAlgorithm import UnlearnAlgorithm
from sast.algorithm.unlearning.FedOSD import FedOSD
from sast.algorithm.unlearning.FedUN import FedUN
from sast.utils.mia import MIAEvaluator

# import backdoors
from sast.dataloaders.backdoors.FigRandBackdoor import FigRandBackdoor

# import dataloader
from sast.dataloaders.separate_data import separate_data, create_data_pool
from sast.dataloaders.DataLoader_cifar10 import DataLoader_cifar10_pat
from sast.dataloaders.DataLoader_mnist import DataLoader_mnist_pat
from sast.dataloaders.DataLoader_fashion import DataLoader_fashion_pat
from sast.dataloaders.DataLoader_cifar100 import DataLoader_cifar100_pat

# import LightSecAgg
from sast.lightsecagg.SecAggMath import SecAggMath
from sast.lightsecagg.FedUN_SecAgg_Client import FedUN_SecAgg_Client
from sast.lightsecagg.FedUN_SecAgg_Server import FedUN_SecAgg_Server

# get the path of the data folder
data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

# get the path of the pool folder
pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)

# get the path of the model
model_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/model/'

# import Task
from sast.utils.Task import BasicTask
from sast.task.Fedunlearning import Fedunlearning
