import torch
import helper
import os
from PIL import Image
import numpy as np

import helper.gan_helper_fun as gh


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, config):
        super(DiffusiveRestoration, self).__init__()
        self.config = config
        self.diffusion = diffusion
        self.device = config.device

        # 判断预训练模型是否存在
        pretrained_model_path = self.config.training.best_resume
        assert os.path.isfile(pretrained_model_path + ".pth.tar"), ('pretrained diffusion model path is wrong!')
        self.diffusion.load_ddm_ckpt(pretrained_model_path, ema=True)
        self.diffusion.model.eval()

    def restore(self, test_loader, r=None):
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                
                if i > 18:

                    # 获取清晰图和带噪图
                    noisy_raw = torch.transpose(sample['noisy_input'],0, 2).squeeze(2).to(self.device)
                    clean_raw = torch.transpose(sample['gt_label_nobias'],0, 2).squeeze(2).to(self.device)

                    image_folder = os.path.join(self.config.data.test_save_dir, str(i))
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)

                    cond = noisy_raw 
                    cond_pic = cond.cpu().detach().numpy()[0].transpose(1,2,0)[..., 0:3]
                    cond_name = image_folder + f'/cond_image_{i}.jpg'
                    Image.fromarray((np.clip(cond_pic,0,1) * 255).astype(np.uint8)).save(cond_name)

                    print(f"=> starting processing image named {i}")

                    x_output = self.diffusive_restoration(clean_raw, noisy_raw, r=r)
                    print(x_output.shape)

                    gen_pic = x_output.cpu().detach().numpy()[0].transpose(1,2,0)[...,0:3]
                    gen_name = image_folder + f'/gen_image_{i}.jpg'
                    Image.fromarray((np.clip(gen_pic,0,1) * 255).astype(np.uint8)).save(gen_name)

    def diffusive_restoration(self, clear_raw, noisy_raw, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(noisy_raw, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x_output = self.diffusion.sample_image(noisy_raw, clear_raw, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
