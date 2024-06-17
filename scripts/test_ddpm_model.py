import argparse
import os, sys, glob

sys.path.append("../.")
sys.path.append("../data/")
sys.path.append("..")
sys.path.append(r"/raid/qinjiahao/projects/starlight_denoising")
import argparse, json, torchvision
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
import numpy as np
from models import ddpm
import helper.canon_supervised_dataset as dset


def get_dataset():    
    composed_transforms = torchvision.transforms.Compose([dset.ToTensor2(), dset.AddFixedNoise(), dset.RandCrop_gen(shape = (128,128))])
    composed_transforms2 = torchvision.transforms.Compose([dset.ToTensor2(), dset.FixedCrop_gen(shape = (128,128))])

    dataset_list = []
    dataset_list_test = []
    
    filepath_noisy = '../data/paired_data/graybackground_mat/'
    dataset_train_gray = dset.Get_sample_noise_batch_3(filepath_noisy, composed_transforms, fixed_noise = False)
    
    dataset_list.append(dataset_train_gray)
    
    all_files_mat = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[0:40]
    all_files_mat_test = glob.glob('../data/paired_data/stillpairs_mat/*.mat')[40:]

    dataset_train_real = dset.Get_sample_batch_3(all_files_mat, composed_transforms)
    dataset_test_real = dset.Get_sample_batch_3(all_files_mat_test, composed_transforms)
    
    dataset_list.append(dataset_train_real)
    dataset_list_test.append(dataset_test_real)
        
    if len(dataset_list)>1:
        dataset_list = torch.utils.data.ConcatDataset(tuple(dataset_list))
        dataset_list_test = torch.utils.data.ConcatDataset(tuple(dataset_list_test))
    else:
        dataset_list= dataset_list[0]
        dataset_list_test = dataset_list_test[0]

    return dataset_list, dataset_list_test

def config_get():
    parser = argparse.ArgumentParser()
    # 参数配置文件路径
    parser.add_argument("--config", default='/raid/qinjiahao/projects/starlight_denoising/scripts/configs.yml', type=str, required=False, help="Path to the config file")
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
    
    dataset_list, dataset_list_test = get_dataset()
    
    train_loader = torch.utils.data.DataLoader(dataset=dataset_list, 
                                               batch_size=1,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=dataset_list_test, 
                                               batch_size=1,
                                               shuffle=False)
    
    model_path = ""

    # 创建模型
    print("=> creating denoising diffusion model")
    diffusion = ddpm.DenoisingDiffusion(config)
    diffusion.test_sample(test_loader, model_path)


if __name__ == "__main__":
    main()
