import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from dataset import Data
from models import DenoisingDiffusion


def config_get():
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='/root/workspace/cgh_workspace/four/diffusion_stamp_C/configs.yml', type=str, required=False, help="Path to the config file")
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    config = config_get()

    # 判断是否使用 cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("=> using device: {}".format(device))
    config.device = device

    # 随机种子
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # 加载数据
    DATASET = Data(config)
    _, val_loader = DATASET.get_loaders()

    # 创建模型
    print("=> creating denoising diffusion model")
    diffusion = DenoisingDiffusion(config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
