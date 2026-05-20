
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta        = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha       = 1 - beta
    alpha_bar   = torch.cumsum(torch.log(alpha), dim=0).exp()
    sigma       = torch.sqrt(beta)

    return {
        "alpha": alpha,      # \alpha_t
        "alpha_bar": alpha_bar,  # \bar{\alpha_t}
        "beta": beta,        # \beta (will be used as \sigma_t^2)
        "sigma": sigma,      # \sigma
    }

# ──────────────────────────────────────────────────────────────
# 替换 load_MNIST → load_GameScene
# 目录结构：multigames_datasets/
#              ├── caveflyer/   (label 0)
#              ├── coinrun/     (label 1)
#              └── starpilot/   (label 2)
# ImageFolder 按文件夹字母序自动分配 label
# Resize 到 28×28 以兼容 UNet 中 AvgPool2d(7) 与 ConvTranspose2d(7,7)
# ──────────────────────────────────────────────────────────────
def load_GameScene(data_root='./multigames_datasets', batch_size=128):
    tf = transforms.Compose([
        transforms.Resize((28, 28)),                              # 保持空间尺寸与UNet兼容
        transforms.ToTensor(),                                    # → (3, 28, 28), [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 仅扩展通道数，数值不变
    ])
    dataset = ImageFolder(root=data_root, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"[Dataset] 共 {len(dataset)} 张图, 类别映射: {dataset.class_to_idx}")
    return dataloader
