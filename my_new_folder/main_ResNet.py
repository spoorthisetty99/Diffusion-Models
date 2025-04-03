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
from model_ResNet import ResNetDiffusion  # Updated to use a ResNet-based model
from score.both import get_inception_and_fid_score

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# ResNet parameters
flags.DEFINE_integer('num_channels', 3, help='number of input image channels')
# UNet-specific parameters
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 4, 8], help='channel multipliers for the model')
flags.DEFINE_multi_integer('attn', [1], help='levels to add attention')
flags.DEFINE_integer('num_res_blocks', 2, help='number of resblocks in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblocks')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help='gradient norm clipping')
flags.DEFINE_integer('total_steps', 40000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 500, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 16, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='number of workers for DataLoader')
flags.DEFINE_float('ema_decay', 0.9999, help='ema decay rate')
flags.DEFINE_bool('parallel', False, help='multi-GPU training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 32, help='sampling size of images')
flags.DEFINE_integer('sample_step', 200, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 500, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 10000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on GPU')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0')

# Placeholder for ema_sampler, replace with your actual sampling function.
def ema_sampler(x_T):
    # This function should generate samples using the EMA model.
    # For demonstration purposes, we simply return x_T.
    return x_T

def train():
    # Set up dataset and dataloader.
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True
    )
    # Create an iterator for the dataloader.
    datalooper = iter(dataloader)
    
    # Initialize ResNet-based diffusion model.
    net_model = ResNetDiffusion(T=FLAGS.T, num_channels=FLAGS.num_channels).to(device)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min(step, FLAGS.warmup) / FLAGS.warmup)
    
    # Trainer wraps the diffusion process (noise scheduling, etc.)
    trainer = GaussianDiffusionTrainer(net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    
    # --- Log Setup ---
    # Create sample directory (use exist_ok to avoid FileExistsError).
    os.makedirs(os.path.join(FLAGS.logdir, 'sample_ResNet1'), exist_ok=True)
    writer = SummaryWriter(FLAGS.logdir)
    
    # Prepare a fixed noise tensor for sampling.
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size).to(device)
    
    # Optionally log a grid of real images from the dataset:
    real_grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer.add_image('real_sample', real_grid)
    writer.flush()
    
    # Show model size.
    model_size = sum(param.data.nelement() for param in net_model.parameters())
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    # --- Training Loop ---
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            # Get a batch of images.
            try:
                x_0 = next(datalooper)
            except StopIteration:
                datalooper = iter(dataloader)
                x_0 = next(datalooper)
            x_0 = x_0[0].to(device)  # CIFAR10 returns (images, labels)

            # Generate noise and add it to the image.
            noise = torch.randn_like(x_0)
            x_noisy = x_0 + noise  # Note: ideally, follow your diffusion schedule here.

            # Sample random timesteps for each image.
            t = torch.randint(0, FLAGS.T, (x_0.shape[0],)).to(device)

            # Model predicts noise from the noisy image.
            predicted_noise = net_model(x_noisy, t)
                
            # Compute mean squared error loss.
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            torch.cuda.empty_cache()  # Clear cached memory to reduce fragmentation

            writer.add_scalar('loss', loss.item(), step)
            pbar.set_postfix(loss='%.3f' % loss.item())

            # Sampling: Save generated sample every sample_step iterations.
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    # Generate samples using your EMA sampler.
                    x_sample = ema_sampler(x_T)
                    grid = (make_grid(x_sample) + 1) / 2
                    sample_path = os.path.join(FLAGS.logdir, 'sample_ResNet1', f'{step}.png')
                    save_image(grid, sample_path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # Saving checkpoint.
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
                
            # Optionally add evaluation here if FLAGS.eval_step > 0.
            
    writer.close()

def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()

if __name__ == '__main__':
    app.run(main)
