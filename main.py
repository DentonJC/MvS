import argparse
from pprint import pprint

import numpy as np
import torch
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import as_multitask
from avalanche.training.supervised import (AGEM, ER_ACE, GEM, MIR, GDumb,
                                           JointTraining, Naive)
from der import DER
from avalanche.training.storage_policy import ClassBalancedBuffer
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

from utils import set_seed
from data import load_dataset
from torchvision.models import mobilenet_v3_small, mobilenet_v2, resnet18
from models import MLP, create_model, CNN, CNNBN, CNNBL, SlimResNet18, return_hidden_resnet18
from mir import MIRPlugin
from replay import Replay
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from avalanche.training.plugins import ReplayPlugin
from gss import GSS_greedyPlugin


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        default="cifar10",
        choices=["cifar10", "cifar100", "img", "fmnist", "mnist", "nette", "cub200"],
    )
    parser.add_argument("--model", default="cnn", choices=["resnet18", "mlp", "mobilenet", "cnn", "cnnbn", "cnnbl", "mobilenet_pretrained", "resnet18_pretrained"])
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--steps", default="one", choices=["one","two","reverse"])
    parser.add_argument("--mode", default="old", choices=["old","new"])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training",
    )
    parser.add_argument(
        "--batch_size_mem",
        type=int,
        default=32,
        help="memory batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=3, help="number of random seeds")
    parser.add_argument("--memory_size", type=int, default=1000, help="replay memory size")
    parser.add_argument("--subsample", type=int, default=50, help='MIR method parameter')
    parser.add_argument(
        "--strategy",
        default="er",
        choices=["er", "der", "gem", "agem", "ace", "mir", "gss"],
        help="Choose the replay strategy",
    )
    parser.add_argument("--external", default='none', choices=["none", "pretrained", "random", "hashes"])
    parser.add_argument("--online", type=bool, default=False)
    parser.add_argument("--remove_current", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.1, help='DER method parameter')
    parser.add_argument("--beta", type=float, default=0.5, help='DER method parameter')
    parser.add_argument("--ti", type=bool, default=False, help='task incremental')
    parser.add_argument("--mem_strength", type=int, default=5)
    parser.add_argument(
        "--selection_strategy",
        default="random",
        choices=["mir", "entropy_min", "entropy_max", "confidence_min", "confidence_max", "margin_min", "margin_max", "kmeans", "coreset", "random", "bayesian_min", "bayesian_max"],
        help="Choose the replay strategy",
    )

    return parser.parse_args()


def test(model, loader, device, args):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for images, labels, task_id in loader:
            images = images.to(device)
            if args.ti:
                pred = model(images, task_id)
            else:
                pred = model(images)
            pred = torch.max(pred.data, 1)[1].cpu()
            correct += (pred == labels.data).sum().numpy()
            total += labels.size(0)
    model.train()
    return correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model(args):
    if args.dataset in ['cifar10']:
        n_classes = 10
        n_tasks = 5
        shape = (3, 32, 32)
    elif args.dataset in ['img']:
        n_classes = 200
        n_tasks = 10
        shape = (3, 64, 64)
    elif args.dataset in ['cub200']:
        n_classes = 200
        n_tasks = 5
        shape = (3, 224, 224)
    elif args.dataset in ['nette']:
        n_classes = 10
        n_tasks = 5
        shape = (3, 224, 224)
    elif 'mnist' in args.dataset:
        n_classes = 10
        n_tasks = 5
        shape = (1, 28, 28)
    else:
        n_classes = 100
        n_tasks = 5
        shape = (3, 32, 32)

    if args.model == "mlp":
        model = MLP(input_size=28 * 28 * shape[0], output_size=n_classes)
    elif args.model == 'cnn':
        model = CNN(n_classes=n_classes)
    elif args.model == 'cnnbn':
        model = CNNBN(n_classes=n_classes)
    elif args.model == 'cnnbl':
        model = CNNBL(n_classes=n_classes)
    elif args.model == 'resnet18':
        model = resnet18(pretrained=False)
        if shape[1] < 224:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, n_classes)    
        model.return_hidden = return_hidden_resnet18.__get__(model)
    elif args.model == 'resnet18_pretrained':
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        model.return_hidden = return_hidden_resnet18.__get__(model)
    elif args.model == 'mobilenet':
        model = mobilenet_v3_small(pretrained=False)
        #model.classifier = nn.Linear(model.classifier.in_features, n_classes)
        if shape[1] < 224:
            model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=n_classes),
        )
    elif args.model == 'mobilenet_pretrained':
        model = mobilenet_v3_small(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=n_classes),
        )

    else:
        raise ValueError(f"Unknown model type: {args.model}")


    print('Params count:', count_parameters(model))
    model.eval()
    input = torch.randn(1, *shape)
    flops = FlopCountAnalysis(model, input)
    print(f"FLOPs: {flops.total()}")

    if args.ti:
        try:
            model = as_multitask(model, "classifier")
        except:
            model = as_multitask(model, "linear")

    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer, n_tasks


def main(args):
    if args.batch_size_mem == 0:
        args.batch_size_mem = args.memory_size
    if args.batch_size_mem < 0:
        args.batch_size_mem = args.batch_size
    pprint(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs = []
    current_accs = []

    for iteration in range(args.trials):
        print("iteration:", iteration)
        set_seed(iteration)
        scenario = load_dataset(args, iteration)

        model, optimizer, n_tasks = init_model(args)

        storage_policy = ClassBalancedBuffer(args.memory_size, adaptive_size=True)
        eval_plugin = EvaluationPlugin(
                     accuracy_metrics(minibatch=False, epoch=False, epoch_running=False, 
                     experience=True, stream=True))
        
        if args.strategy == "der":
            cl_strategy = DER(
                model=model,
                optimizer=optimizer,
                device=device,
                batch_size_mem=args.batch_size_mem,
                train_mb_size=args.batch_size,
                eval_mb_size=64,
                train_epochs=args.epochs,
                mem_size=args.memory_size,
                evaluator=eval_plugin,
                remove_current=args.remove_current,
                args=args,
                ti=args.ti,
                selection_strategy=args.selection_strategy,
                steps=args.steps,
                mode=args.mode,
                alpha=args.alpha,
                beta=args.beta,
            )
               
        elif args.strategy == "ace":
            cl_strategy = ER_ACE(
                model=model,
                optimizer=optimizer,
                plugins=[],
                device=device,
                train_mb_size=args.batch_size,
                eval_mb_size=64,
                mem_size=args.memory_size,
                train_epochs=args.epochs,
                evaluator=eval_plugin,
            )
        elif args.strategy == "mir":
            cl_strategy = Replay(
                model=model,
                optimizer=optimizer,
                device=device,
                batch_size_mem=args.batch_size_mem,
                train_mb_size=args.batch_size,
                eval_mb_size=64,
                train_epochs=args.epochs,
                mem_size=args.memory_size,
                evaluator=eval_plugin,
                remove_current=args.remove_current,
                args=args,
                ti=args.ti,
                selection_strategy=args.selection_strategy,
                steps=args.steps,
                mode=args.mode,                    
                plugins=[MIRPlugin(mem_size=args.memory_size, subsample=args.subsample, batch_size_mem=args.batch_size_mem)],
            )
        elif args.strategy == "gss":
            input_size = [3,32,32]
            if args.dataset in ['nette','cub200']:
                input_size = [3,244,244]
            if args.dataset in ['img']:
                input_size = [3,64,64]
            cl_strategy = Replay(
                model=model,
                optimizer=optimizer,
                device=device,
                batch_size_mem=args.batch_size_mem,
                train_mb_size=args.batch_size,
                eval_mb_size=64,
                train_epochs=args.epochs,
                mem_size=args.memory_size,
                evaluator=eval_plugin,
                remove_current=args.remove_current,
                ti=args.ti,
                plugins=[GSS_greedyPlugin(mem_size=args.memory_size, input_size=input_size, mem_strength=args.mem_strength)],
            )
        else:
            cl_strategy = Replay(
                model=model,
                optimizer=optimizer,
                device=device,
                batch_size_mem=args.batch_size_mem,
                train_mb_size=args.batch_size,
                eval_mb_size=64,
                train_epochs=args.epochs,
                mem_size=args.memory_size,
                evaluator=eval_plugin,
                remove_current=args.remove_current,
                args=args,
                ti=args.ti,
                selection_strategy=args.selection_strategy,
                steps=args.steps,
                mode=args.mode,
            )

        if args.online:
            train_stream = split_online_stream(
                original_stream=scenario.train_stream,
                experience_size=args.batch_size,
                access_task_boundaries=args.ti,
            )
            for task, experience in tqdm(enumerate(train_stream)):
                cl_strategy.train(
                    experience,
                    eval_streams=[],
                    num_workers=0,
                    drop_last=True,
                )
    
            task_accs = []
            for i in range(n_tasks):
                test_loader = DataLoader(
                    scenario.test_stream[i].dataset,
                    batch_size=256,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=1,
                )
                test_acc = test(cl_strategy.model, test_loader, device, args)
                task_accs.append(test_acc)
                print('Task ' + str(i) + ':', test_acc)
            print("task_accs:", task_accs, np.mean(task_accs))
        else:
            train_stream = scenario.train_stream
            last_accs = []
            for task, experience in enumerate(train_stream):
                cl_strategy.train(
                    experience,
                    eval_streams=[],
                    num_workers=0,
                    drop_last=True,
                )
    
                task_accs = []
                for i in range(task+1):
                    test_loader = DataLoader(
                        scenario.test_stream[i].dataset,
                        batch_size=256,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=1,
                    )
                    task_accs.append(test(cl_strategy.model, test_loader, device, args))
                last_accs.append(task_accs[-1])
                print("task_accs:", task_accs, np.mean(task_accs))
        accs.append(np.mean(task_accs))
        if args.online:
            current_accs.append(task_accs[-1])
        else:
            print('last_accs:', last_accs)
            current_accs.append(np.mean(last_accs))

        print("accs:", accs)
        print("accuracy:", np.mean(accs))
        print("std:", np.std(accs))
        print("acts:", current_accs)
        print("acta:", np.mean(current_accs))
        print("acta_std:", np.std(current_accs))
        
    return np.mean(accs)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
