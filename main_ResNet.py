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
from model_ResNet import ResNetDenoiser  # Import the new ResNet-based denoiser
from score.both import get_inception_and_fid_score

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')

# Model parameters (used for ResNetDenoiser)
flags.DEFINE_integer('ch', 128, help='base channel count (used as the ResNet backbone channels)')
flags.DEFINE_float('dropout', 0.1, help='dropout rate (if applicable)')

# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')

# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 40000, help='total training steps')
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

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        # Ensure both source and target tensors are on the same device
        target_dict[key].data.copy_(
            target_dict[key].data.to(device) * decay +
            source_dict[key].data.to(device) * (1 - decay)
        )

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "Generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size)).to(device)
            batch_images = sampler(x_T).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def train():
    # Dataset
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
    datalooper = infiniteloop(dataloader)

    # Model setup: move both models to the device.
    # Instantiate the new ResNetDenoiser.
    net_model = ResNetDenoiser(
        T=FLAGS.T, base_channels=FLAGS.ch
    ).to(device)
    ema_model = copy.deepcopy(net_model).to(device)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T
    ).to(device)
    
    # Instantiate samplers for the reverse diffusion process.
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type
    ).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type
    ).to(device)

    # Create results folder.
    os.makedirs(os.path.join(FLAGS.logdir, 'sample_ResNetDenoiser'), exist_ok=True)

    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size).to(device)
    writer = SummaryWriter(FLAGS.logdir)

    # Training loop.
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # Save samples using the EMA sampler for denoising.
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0_denoised = ema_sampler(x_T)
                    grid = (make_grid(x_0_denoised) + 1) / 2
                    save_path = os.path.join(FLAGS.logdir, 'sample_ResNetDenoiser', f'{step}.png')
                    save_image(grid, save_path)
                net_model.train()

            # Save model checkpoint.
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                torch.save(net_model.state_dict(), os.path.join(FLAGS.logdir, 'ckpt_ResNetDenoiser.pt'))

    writer.close()

def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    elif FLAGS.eval:
        pass  # Implement evaluation if needed
    else:
        print('Add --train and/or --eval to execute corresponding tasks')

if __name__ == '__main__':
    app.run(main)
