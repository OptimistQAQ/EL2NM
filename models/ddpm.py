import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import helper
from models.unet import DiffusionUNet, Gauss_Forward
from PIL import Image

import helper.canon_supervised_dataset as dset
import helper.gan_helper_fun as gh

from torchvision.transforms.functional import crop

from torch.utils.tensorboard import SummaryWriter

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# 用于实现指数移动平均(Exponential Moving Average, EMA)方法来平滑模型参数的更新，以提高模型的泛化能力。
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data.to('cuda')


    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

# 获取一组时间序列的数组
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# 根据输入的图像序列和噪声，预测噪声，并计算噪声的估计损失
def noise_estimation_loss(model, x0, t, e, b):
    # 连乘，通过公式计算，一步将图片添加为高斯噪声
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, :3, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    
    # x_cond = inverse_data_transform(x)
    # cond = gh.split_into_patches2d(x_cond, patch_size=128)
    # cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
    # cond_name = f'./output_x.jpg'
    # Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)
    
    output = model(torch.cat([x0[:, 3:, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)
        # self.model=nn.DataParallel(self.model,device_ids=[1])
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)
        self.num_timesteps = 50

        self.optimizer = helper.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.dtype = torch.float32
        
        self.forward_process = Gauss_Forward(num_timesteps=self.num_timesteps, device=self.device)

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        # self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = helper.logging.load_checkpoint(load_path + ".pth.tar", None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, train_loader, test_loader):
        cudnn.benchmark = True

        if os.path.isfile(self.config.training.best_resume + ".pth.tar"):
            self.load_ddm_ckpt(self.config.training.best_resume)
        
        best_loss = 10000
            
        writer = SummaryWriter("./logs_3")

        # 训练
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, sample in enumerate(train_loader):
                # 获取清晰图和带噪图
                noisy_raw = torch.transpose(sample['noisy_input'],0, 2).squeeze(2).to(self.device)
                clean_raw = torch.transpose(sample['gt_label_nobias'],0, 2).squeeze(2).to(self.device)

                # 将两张图拼接
                x = torch.cat([clean_raw, noisy_raw], dim=1)

                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                # 生成一个随机正态分布的e
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling t.shape=[9]
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                # t.shape= [16]
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                x_gauss = clean_raw.clone()
                
                for j in range(self.num_timesteps):
                    x_gauss = self.forward_process.forward(x_gauss, j)

                out_put = self.model(torch.cat([clean_raw, x_gauss], dim=1), t.float())

                cond = gh.split_into_patches2d(out_put, patch_size=128)
                cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
                cond_name = f'./output.jpg'
                Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

                loss = (noisy_raw - out_put).square().sum(dim=(1, 2, 3)).mean(dim=0)

                output_noisy = gh.split_into_patches2d(noisy_raw, patch_size=128)
                cond_pic = output_noisy.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
                cond_name = f'./output_noisy.jpg'
                Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

                # loss = noise_estimation_loss(self.model, x, t, e, b)
                writer.add_scalar("loss", loss, self.step)

                if self.step % 10 == 0:
                    print('epoch: %d, i: %d, step: %d, loss: %.6f, time consumption: %.6f' % (epoch, i, self.step, loss.item(), data_time / (i+1)))

                # 更新参数
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
                # self.sample_validation_patches(test_loader, self.step)
                # self.sample(test_loader, self.step, t)

                # 模型测试
                if self.step % 1000 == 0:
                    self.model.eval()
                    # self.sample_validation_patches(test_loader, self.step)
                    self.sample(test_loader, self.step, t)

                # save best models
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    helper.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'config': self.config
                    }, filename=self.config.training.best_resume)

                # save models
                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    helper.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'config': self.config
                    }, filename=self.config.training.resume)

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.sampling.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        
        output = torch.zeros_like(x_cond, device=x.device)
        
        t = torch.randint(low=0, high=self.num_timesteps, size=(16 // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:16]
        
        for (hi, wi) in patch_locs:
            x_patch = crop(x, hi, wi, patch_size, patch_size)
            x_cond_patch = crop(x_cond, hi, wi, patch_size, patch_size)
            
            out_patch = self.test(x_cond_patch, x_patch, t)
            
            output[:, :, hi:hi+patch_size, wi:wi+patch_size] = out_patch
        return output
    

    def test_sample(self, test_loader, model_path):

        if os.path.isfile(model_path):
            self.load_ddm_ckpt(model_path)
        
        n = 16
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

        self.sample(test_loader, self.step, t)
        


    def sample(self, val_loader, step, t):
        with torch.no_grad():
            total_kld = 0
            print(f"Processing a single batch of validation images at step: {step}")
            for i, sample in enumerate(val_loader):
                
                noisy_raw = torch.transpose(sample['noisy_input'], 0, 2).squeeze(2).to(self.device)
                clean_raw = torch.transpose(sample['gt_label_nobias'], 0, 2).squeeze(2).to(self.device)

                gauss = clean_raw.clone()
                for j in range(self.num_timesteps):
                    gauss = self.forward_process.forward(gauss, j)

                direct_out = None

                h = self.num_timesteps - 30
                while(h):
                    step = torch.full((16,), self.num_timesteps - 10, dtype=torch.long).cuda()
                    out_put, cur_direct_out = self.sample_one_step(clean_raw, gauss, step, t)
                    if direct_out is None:
                        direct_out = cur_direct_out
                    gauss = out_put
                    h = h-1

                # x = torch.randn(noisy_raw.shape, device=self.device)

                # out_put = self.model(torch.cat([noisy_raw, clean_raw], dim=1), t.float())
                
                # 存到指定文件夹中
                image_folder = os.path.join("./val/", str(self.step))
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)

                cond = gh.split_into_patches2d(noisy_raw, patch_size=128)
                cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
                cond_name = image_folder + f'/test_output_noisy_{i}.jpg'
                Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

                clean = gh.split_into_patches2d(clean_raw, patch_size=128)
                clean_pic = clean.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
                clean_name = image_folder + f'/test_output_clean_{i}.jpg'
                Image.fromarray((np.clip(clean_pic,0,1) * 255).astype(np.uint8)).save(clean_name)

                output = gh.split_into_patches2d(out_put, patch_size=128)
                out_pic = output.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
                cond_name = image_folder + f'/test_output_{i}.jpg'
                Image.fromarray((np.clip(out_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

                directput = gh.split_into_patches2d(direct_out, patch_size=128)
                directout_pic = directput.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
                cond_name = image_folder + f'/test_directput_{i}.jpg'
                Image.fromarray((np.clip(directout_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

                kld_val = gh.cal_kld(directout_pic, cond_pic)
                total_kld += kld_val
                print("========================>i: {}, kld_val: {}".format(i, kld_val))
                
            print("Total KLD value: ", total_kld)
            print("Average KLD value: ", total_kld/len(val_loader))

    # 添加一个输出噪声图的函数            
    def test(self, noisy_raw, clean_raw, t):
        gauss = clean_raw.clone()
        for j in range(self.num_timesteps):
            gauss = self.forward_process.forward(gauss, j)
        direct_out = None

        h = self.num_timesteps - 27
        while(h):
            step = torch.full((16,), self.num_timesteps - 10, dtype=torch.long).cuda()
            out_put, cur_direct_out = self.sample_one_step(clean_raw, gauss, step, t)
            if direct_out is None:
                direct_out = cur_direct_out
            gauss = out_put
            h = h-1

        return out_put

    @torch.no_grad()    
    def sample_one_step(self, noisy_raw, gauss, t, step):
        
        out_put = self.model(torch.cat([noisy_raw, gauss], dim=1), step.float())

        direct_out = out_put.clone()
        
        x_times = out_put.clone()
        x_times_sub = x_times.clone()
        
        cur_time = torch.zeros_like(t)
        fp_index = torch.where(cur_time < t)[0]
        for i in range(t.max()):
            x_times_sub = x_times.clone()
            x_times[fp_index] = self.forward_process.forward(x_times[fp_index], i)
            cur_time += 1
            fp_index = torch.where(cur_time < t)[0]

        out = gauss - x_times + x_times_sub
        
        out = torch.clip(out, 0, 1)

        # cond = gh.split_into_patches2d(gauss, patch_size=128)
        # cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
        # cond_name = f'./inference/gauss{self.step}.jpg'
        # Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)
        
        # cond = gh.split_into_patches2d(x_times, patch_size=128)
        # cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
        # cond_name = f'./inference/x_times{self.step}.jpg'
        # Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

        # cond = gh.split_into_patches2d(x_times_sub, patch_size=128)
        # cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
        # cond_name = f'./inference/x_times_sub{self.step}.jpg'
        # Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

        # cond = gh.split_into_patches2d(out, patch_size=128)
        # cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
        # cond_name = f'./inference/out{self.step}.jpg'
        # Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)
        return out, direct_out

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.config.data.val_save_dir, str(self.config.data.image_size))
        with torch.no_grad():
            # 添加一张测试图
            print(f"Processing a single batch of validation images at step: {step}")
            for i, sample in enumerate(val_loader):
                noisy_raw = torch.transpose(sample['noisy_input'],0, 2).squeeze(2).to(self.device)
                clean_raw = torch.transpose(sample['gt_label_nobias'],0, 2).squeeze(2).to(self.device)
                x = torch.cat([clean_raw, noisy_raw], dim=1)
                break
            n = x.size(0)
            # 获取后面四通道的条件图
            x_cond = x[:, 3:6, :, :].to(self.device)  # 条件图像
            x_cond = data_transform(x_cond)
            
            # 存到指定文件夹中
            image_folder = os.path.join(self.config.data.val_save_dir, str(self.step))
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            
            # x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = x[:, :3, :, :].to(self.device)
            
            # x = data_transform(x)
            # 生成一张图片
            xxx = self.sample_image(x_cond, x)
            xx = xxx[1]
            x0 = xxx[0]
            
            # 保存参考图
            x_cond = inverse_data_transform(x_cond)
            cond = gh.split_into_patches2d(x_cond, patch_size=128)
            cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
            cond_name = image_folder + f'/cond_image_{self.step}.jpg'
            Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)
            
            # 保存生成的图
            for i in range(0, 26):
                x = x0[i]
                # x = inverse_data_transform(x)
                gen = gh.split_into_patches2d(x, patch_size=128)   
                gen_pic = gen.cpu().detach().numpy()[0].transpose(1,2,0)[...,0:3]
                gen_name = image_folder + f'/gen_image_{self.step}_{i}.jpg'
                Image.fromarray((np.clip(gen_pic,0,1) * 255).astype(np.uint8)).save(gen_name)
            
            for i in range(0, 25):
                x = xx[i]
                # x = inverse_data_transform(x)
                gen = gh.split_into_patches2d(x, patch_size=128)   
                gen_pic = gen.cpu().detach().numpy()[0].transpose(1,2,0)
                gen_name = image_folder + f'/gen_image_{self.step}_{i+26}.png'
                Image.fromarray((np.clip(gen_pic,0,1) * 255).astype(np.uint8)).save(gen_name)
