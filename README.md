#  EL2NM: Extremely Low-light Noise Modeling Through Diffusion Iteration 
> Official implementation of EL2NM: Extremely Low-light Noise Modeling Through Diffusion Iteration in pytorch.
> [[HomePage\]](https://optimistqaq.github.io/Page_EL2NM/) [[Paper\]](https://openaccess.thecvf.com/content/CVPR2024W/MIPI/papers/Qin_EL2NM_Extremely_Low-light_Noise_Modeling_Through_Diffusion_Iteration_CVPRW_2024_paper.pdf) [[Dataset\]](https://kristinamonakhova.com/starlight_denoising/#dataset) [[Checkpoints\]](https://drive.google.com/drive/folders/1s6VsoIkWQvuZQqMsxPlV7crXEUOrvASK?usp=sharing)

## 🎉 News

***(2024.03.28)\***: 🎉 Our paper was accepted by *IEEE/CVF International Conference on Computer Vision Workshops* (CVPRW) 2024~~



## 📋 Prerequisites



- Python >=3.6, PyTorch >= 1.6
- Requirements: opencv-python, pandas, scipy, lpips, yacs4
- Platforms: Ubuntu 18.04, cuda-11.3
- Our method can run on the CPU, but we recommend you run it on the GPU



## 🎬 Quick Start



1. Clone this project using:

```python
git clone https://github.com/OptimistQAQ/EL2NM.git
```

2. Install the dependencies using:

```yaml
conda env create -f environment.yml
source activate el2nm
```

3. Training using:

```bash
cd scripts
python train_ddpm_model.py
```

4. Testing using:

```bash
cd scripts
python test_ddpm_model.py
```



## 🏷️ Citation



If you find our code helpful in your research or work please cite our paper.

```powershell
@InProceedings{Qin_2024_CVPR,
    author    = {Qin, Jiahao and Qin, Pinle and Chai, Rui and Qin, Jia and Jin, Zanxia},
    title     = {EL2NM: Extremely Low-light Noise Modeling Through Diffusion Iteration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {1085-1094}
}
```



## 🤝 Acknowledgments



 Part of the code comes from other repo, please abide by the original open source license for the relevant code. 

- [Starlight (CVPR 2022)](https://github.com/monakhova/starlight_denoising)
- [SID (CVPR 2018)](https://github.com/cchen156/Learning-to-See-in-the-Dark)
- [ELD (CVPR 2020 / TPAMI 2021)](https://github.com/Vandermode/ELD)
- [PMN (ACM MM 2022 / TPAMI 2023)](https://github.com/megvii-research/PMN)

