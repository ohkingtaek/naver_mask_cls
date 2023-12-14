import argparse
import os
import multiprocessing
import random
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.mask_dataset import MaskDatasetV1, MaskDatasetV2
from models.mask_model import MaskModelV1, MaskModelV2, MaskModelV3, MaskModelV4
from utils.transform import TrainAugmentation, TestAugmentation
from utils.utils import get_lr
from ops.losses import get_loss

import warnings
warnings.filterwarnings('ignore')

_Optimizer = torch.optim.Optimizer
_Scheduler = torch.optim.lr_scheduler._LRScheduler


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(
    configs: Dict,
    save_dir: str,
    dataloader: DataLoader,
    device: str, model: nn.Module,
    loss_fn: nn.Module,
    optimizer: _Optimizer,
    scheduler: _Scheduler,
    epoch: int
) -> None:
    """데이터셋으로 뉴럴 네트워크를 훈련합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    :param optimizer: 훈련에 사용되는 옵티마이저
    :type optimizer: torch.optim.Optimizer
    """
    model.train()

    loss_value = 0
    mask_loss_value = 0
    gender_loss_value = 0
    age_loss_value = 0

    matches = 0
    mask_matches = 0
    gender_matches = 0
    age_matches = 0

    epochs = configs['train']['epoch']

    for batch, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.float().to(device)
        # target1, target2, target3 = targets
        # target1 = target1.float().to(device)
        # target2 = target2.float().to(device)
        # target3 = target3.float().to(device)

        # output1, output2, output3 = model(images)
        # loss_1 = loss_fn(output1, target1)
        # loss_2 = loss_fn(output2, target2)
        # loss_3 = loss_fn(output3, target3)
        # total_loss = loss_1 + loss_2 + loss_3
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        # total_loss.backward()
        loss.backward()
        optimizer.step()

        # mask_loss_value += loss_1.item()
        # gender_loss_value += loss_2.item()
        # age_loss_value += loss_3.item()
        # loss_value += total_loss.item()
        loss_value += loss.item()

        outputs = F.softmax(outputs, dim=-1)
        matches += (outputs == targets).sum().item()
        # mask_matches += (F.softmax(output1, dim=1) == target1).sum().item()
        # gender_matches += (F.softmax(output2, dim=1) == target2).sum().item()
        # age_matches += (F.softmax(output3, dim=1) == target3).sum().item()

        if (batch+1) % 50 == 0:
            train_loss = loss_value / 50
            # mask_loss = mask_loss_value / 50
            # gender_loss = gender_loss_value / 50
            # age_loss = age_loss_value / 50

            # mask_acc = mask_matches / configs['train']['batch_size'] / 50
            # gender_acc = gender_matches / configs['train']['batch_size'] / 50
            # age_acc = age_matches / configs['train']['batch_size'] / 50
            # train_acc = (mask_acc + gender_acc + age_acc) / 3
            train_acc = matches / configs['train']['batch_size'] / 50

            current_lr = get_lr(optimizer)
            image = images[0, ...].detach().cpu().numpy()
            image = image.transpose(1, 2, 0)

            print(
                f"Epoch[{epoch}/{epochs}]({batch + 1}/{len(dataloader)}) "
                f"| lr {current_lr} \ntrain loss {train_loss:4.4}"
                # f"| mask acc {mask_loss:4.4} | gender loss {gender_loss:4.4} "
                # f"| age loss {age_loss:4.4} \n"
                f"train acc {train_acc:4.2%}"# | mask acc {mask_acc:4.2%}
                # f"| gender acc {gender_acc:4.2%} | age acc {age_acc:4.2%}"
            )
            # wandb.logger.experiment.log({
            #     'train_rgb': wandb.Image(image, caption='Input-image')
            # })

            loss_value = 0
            # mask_loss_value = 0
            # gender_loss_value = 0
            # age_loss_value = 0

            # mask_matches = 0
            # gender_matches = 0
            # age_matches = 0

    if scheduler is not None:
        scheduler.step()
    torch.save(
        model.state_dict(),
        f'{save_dir}/{epoch}-{train_loss:4.4}-{train_acc:4.2}.pth'
        )
    print(
        f'Saved Model State to {save_dir}/{epoch}-{train_loss:4.4}-{train_acc:4.2}.pth'
    )


def validation(
    dataloader: DataLoader,
    device: str,
    model: nn.Module,
    loss_fn: nn.Module,
    epoch: int
) -> None:
    """데이터셋으로 뉴럴 네트워크의 성능을 검증합니다.

    :param dataloader: 파이토치 데이터로더
    :type dataloader: DataLoader
    :param device: 훈련에 사용되는 장치
    :type device: str
    :param model: 훈련에 사용되는 모델
    :type model: nn.Module
    :param loss_fn: 훈련에 사용되는 오차 함수
    :type loss_fn: nn.Module
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    valid_mask_loss = 0
    valid_gender_loss = 0
    valid_age_loss = 0
    valid_loss = 0

    mask_matches = 0
    gender_matches = 0
    age_matches = 0
    matches = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.float().to(device)
            # target1, target2, target3 = targets
            # target1 = target1.float().to(device)
            # target2 = target2.float().to(device)
            # target3 = target3.float().to(device)
            outputs = model(images)
            # outputs = torch.argmax(outputs, dim=-1)
            loss = loss_fn(outputs, targets)

            # output1, output2, output3 = model(images)
            # loss_1 = loss_fn(output1, target1)
            # loss_2 = loss_fn(output2, target2)
            # loss_3 = loss_fn(output3, target3)
            # total_loss = loss_1 + loss_2 + loss_3

            # valid_mask_loss += loss_1.item()
            # valid_gender_loss += loss_2.item()
            # valid_age_loss += loss_3.item()
            # valid_loss += total_loss.item()
            valid_loss += loss.item()

            # mask_matches += (F.softmax(output1, dim=1) == target1).sum().item()
            # gender_matches += (F.softmax(output2, dim=1) == target2).sum().item()
            # age_matches += (F.softmax(output3, dim=1) == target3).sum().item()
            # matches += (mask_matches + gender_matches + age_matches) / 3
            outputs = F.softmax(outputs, dim=-1)
            matches += (outputs == targets).sum().item()

    valid_loss /= num_batches
    # valid_mask_loss /= num_batches
    # valid_gender_loss /= num_batches
    # valid_age_loss /= num_batches

    matches /= size
    # mask_matches /= size
    # gender_matches /= size
    # age_matches /= size
    print(
        f"Epoch[{epoch}]({len(dataloader)})"
        f"valid loss {valid_loss:4.4}"# | mask loss {valid_mask_loss:4.4} "
        # f"| gender loss {valid_gender_loss:4.4} | age loss {valid_age_loss:4.4}"
        f"\nvalid acc {matches:4.2%}"# | mask acc {mask_matches:4.2%} "
        # f"| gender acc {gender_matches:4.2%} | age acc {age_matches:4.2%}"
    )


def run_pytorch(configs) -> None:
    """학습 파이토치 파이프라인

    :param configs: 학습에 사용할 config들
    :type configs: dict
    """
    # wandb_logger = WandbLogger(project="naver_mask_classification")
    # wandb.config.update(configs)
    train_augmentation = TrainAugmentation(resize=[380, 380])
    train_data = MaskDatasetV2(
        image_dir=configs['data']['train_dir'],
        csv_path=configs['data']['csv_dir'],
        transform=train_augmentation,
        valid=False,
        valid_rate=configs['data']['valid_rate']
    )

    valid_augmentation = TestAugmentation(resize=[380, 380])
    val_data = MaskDatasetV2(
        image_dir=configs['data']['train_dir'],
        csv_path=configs['data']['csv_dir'],
        transform=valid_augmentation,
        valid=True,
        valid_rate=configs['data']['valid_rate']
    )

    train_loader = DataLoader(
        train_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=configs['train']['batch_size'],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MaskModelV4(num_classes=18).to(device)

    loss_fn = get_loss()
    optimizer = optim.Adam(model.parameters(), lr=configs['train']['lr'])
    scheduler = None

    save_dir = os.path.join(configs['ckpt_path'], str(model.__class__.__name__))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    i = 0
    while True:
        version = 'v' + str(i)
        if os.path.exists(os.path.join(save_dir, version)):
            if not os.listdir(os.path.join(save_dir, version)):
                save_dir = os.path.join(save_dir, version)
                break
            i += 1
            continue
        else:
            save_dir = os.path.join(save_dir, version)
            os.makedirs(save_dir)
            break

    for e in range(configs['train']['epoch']):
        print(f'Epoch {e+1}\n-------------------------------')
        train(
            configs, save_dir, train_loader,
            device, model, loss_fn, optimizer, scheduler, e+1
        )
        validation(val_loader, device, model, loss_fn, e+1)
        print('\n')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="train.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    seed_everything(configs['seed'])
    run_pytorch(configs=configs)
