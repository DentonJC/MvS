import random
import numpy as np
import torch
from torchvision import transforms

from avalanche.benchmarks.classic import (SplitCIFAR10, SplitCIFAR100,
                                          SplitFMNIST, SplitMNIST,
                                          SplitTinyImageNet, SplitCUB200)
from avalanche.benchmarks.datasets.dataset_utils import \
    default_dataset_location
from avalanche.benchmarks.scenarios.supervised import \
    class_incremental_benchmark
from avalanche.benchmarks.scenarios.task_aware import \
    task_incremental_benchmark
from avalanche.benchmarks.utils import as_classification_dataset
from torchvision.datasets import SVHN
try:
    from avalanche.benchmarks.generators.benchmark_generators import \
        benchmark_with_validation_stream
except:
    from avalanche.benchmarks.scenarios.validation_scenario import (
        benchmark_with_validation_stream,
    )

from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
from imagenette import SplitImagenette


def load_dataset(args, seed=42):
    if args.dataset == "fmnist":
        scenario = SplitFMNIST(
            5,
            return_task_id=args.ti,
            seed=seed,
            fixed_class_order=np.arange(10),
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
        )
    elif args.dataset == "cifar10":
        scenario = SplitCIFAR10(
            5,
            return_task_id=args.ti,
            seed=seed,
            fixed_class_order=np.arange(10),
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
        )
    elif args.dataset == "cifar100":
        scenario = SplitCIFAR100(
            5,
            return_task_id=args.ti,
            seed=seed,
            fixed_class_order=np.arange(100),
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
        )
    elif args.dataset == "cub200":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]) 
        scenario = SplitCUB200(
            5,
            return_task_id=args.ti,
            seed=seed,
            fixed_class_order=np.arange(200),
            shuffle=True,
            train_transform=transform,
            eval_transform=transform,
            class_ids_from_zero_in_each_exp=False,
        )
    elif args.dataset == "mnist":
        scenario = SplitMNIST(
            5,
            return_task_id=args.ti,
            seed=seed,
            fixed_class_order=np.arange(10),
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
        )
    elif args.dataset == "nette":
        scenario = SplitImagenette(
            5,
            return_task_id=args.ti,
            seed=seed,
            fixed_class_order=np.arange(10),
            shuffle=True,
            class_ids_from_zero_in_each_exp=False,
        )
    elif args.dataset == "img":
        if 'pretrained' in args.model:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]) 
            scenario = SplitTinyImageNet(
                10,
                return_task_id=args.ti,
                seed=seed,
                fixed_class_order=np.arange(200),
                train_transform=transform,
                eval_transform=transform,
                shuffle=True,
                class_ids_from_zero_in_each_exp=False,
            )
        else:
            scenario = SplitTinyImageNet(
                10,
                return_task_id=args.ti,
                seed=seed,
                fixed_class_order=np.arange(200),
                shuffle=True,
                class_ids_from_zero_in_each_exp=False,
            )
    else:
        pass

    return scenario
