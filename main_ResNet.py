import copy
import json
import os
import warnings

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import ResNetDiffusion  # Updated to use ResNet instead of U-Net
from score.both import get_inception_and_fid_score

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# ResNet parameters
flags.DEFINE_integer('num_channels', 3, help='number of input image channels')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0')

def train():
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)

    # Initialize ResNet model
    net_model = ResNetDiffusion(T=FLAGS.T, num_channels=FLAGS.num_channels).to(device)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min(step, FLAGS.warmup) / FLAGS.warmup)
    trainer = GaussianDiffusionTrainer(net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(FLAGS.logdir)
    
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x_0 = next(iter(dataloader))[0].to(device)
            
            # Generate noisy image
            noise = torch.randn_like(x_0)
            x_noisy = x_0 + noise
            
            # Predict noise and denoise
            estimated_noise, denoised_x = net_model(x_noisy, torch.randint(0, FLAGS.T, (x_0.shape[0],)).to(device))
            
            # Compute losses
            noise_loss = torch.nn.functional.mse_loss(estimated_noise, noise)
            denoise_loss = torch.nn.functional.mse_loss(denoised_x, x_0)
            total_loss = noise_loss + denoise_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()

            writer.add_scalar('loss', total_loss, step)
            pbar.set_postfix(loss='%.3f' % total_loss.item())
    writer.close()

def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()

if __name__ == '__main__':
    app.run(main)
