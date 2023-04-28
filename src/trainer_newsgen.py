# Copyright 2023 The Newsgen Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import torch
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from newsgen.base_model import NewsgenBase
from data import EncodedVisualNewsDataset

parser = argparse.ArgumentParser()
parser.add_argument('--encoded_dataset',
                    default='./data/visual_news/encoded_headlines.json',
                    help='Json file containing encoded VisualNews dataset')
parser.add_argument('--log_dir',
                    default='./logs',
                    help='Directory to save logs')
parser.add_argument('--exp_name',
                    default='newsgen_exp1',
                    help='Experiment name to use to save logs')
parser.add_argument('--ckpt_dir',
                    default='./checkpoints/newsgen',
                    help='Directory to save checkpoints')
parser.add_argument('--hparams',
                    default='./newsgen/hparams_base.json',
                    help='Path to hparams json file')
parser.add_argument('--vqgan_ckpt',
                    default='./pretrained/vqgan.ckpt',
                    help='Path to VQGAN checkpoint')
parser.add_argument('--vqgan_hparams',
                    default='./hparams_vqgan.json',
                    help='Path to vqgan hparams json file')
parser.add_argument('--warmup_steps',
                    default=4000,
                    type=int,
                    help='LR scheduler warmup steps')
parser.add_argument('--total_steps',
                    default=40000,
                    type=int,
                    help='Total training steps')
parser.add_argument('--train_batch',
                    default=8,
                    type=int,
                    help='Train batch size')
parser.add_argument('--val_batch',
                    default=8,
                    type=int,
                    help='Validation batch size')
parser.add_argument('--test_batch', default=8, type=int, help='Test batch size')
parser.add_argument('--num_workers',
                    default=8,
                    type=int,
                    help='Number of dataloader workers to use')
parser.add_argument('--accelerator',
                    default='auto',
                    help='Accelerator to use for training')
parser.add_argument('--strategy',
                    default='auto',
                    help='Strategy to use for training')
parser.add_argument('--distributed',
                    default=True,
                    type=bool,
                    help='If distributed training, use DistributedSampler')
parser.add_argument('--epochs',
                    default=5,
                    type=int,
                    help='Number of epochs to train')
parser.add_argument('--val_every_n_steps',
                    default=3000,
                    type=int,
                    help='Perform validation every n steps and save checkpoint')
parser.add_argument('--log_every_n_steps',
                    default=1000,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--seed', default=123, type=int, help='Random seed to use')


def main():
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    seed_everything(args.seed, workers=True)

    # Load hparams
    assert os.path.exists(args.hparams), f'{args.hparams} does not exist.'
    assert os.path.isfile(args.hparams), f'{args.hparams} is not a file.'

    with open(args.hparams, 'r') as f:
        hparams = json.load(f)

    # Load dataset
    assert os.path.exists(
        args.encoded_dataset), f'{args.encoded_dataset} does not exist.'
    assert os.path.isfile(
        args.encoded_dataset), f'{args.encoded_dataset} is not a file.'

    train_set = EncodedVisualNewsDataset(args.encoded_dataset, 'train')
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch,
                              shuffle=(not args.distributed),
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_set = EncodedVisualNewsDataset(args.encoded_dataset, 'val')
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch,
                            shuffle=(not args.distributed),
                            num_workers=args.num_workers,
                            pin_memory=True)

    test_set = EncodedVisualNewsDataset(args.encoded_dataset, 'test')
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch,
                             shuffle=(not args.distributed),
                             num_workers=args.num_workers,
                             pin_memory=True)

    # Define logger and checkpoint callback
    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='{epoch:02d}-{step:06d}',
        save_last=True,
        save_top_k=-1)

    # Learning rate scheduler config
    hparams['warmup_steps'] = args.warmup_steps
    hparams['training_steps'] = args.total_steps

    # Load vqgan hparams
    assert os.path.exists(
        args.vqgan_hparams), f'{args.vqgan_hparams} does not exist.'
    assert os.path.isfile(
        args.vqgan_hparams), f'{args.vqgan_hparams} is not a file.'

    with open(args.vqgan_hparams, 'r') as f:
        vqgan_hparams = json.load(f)

    # Load model
    assert os.path.exists(args.vqgan_ckpt), f'{args.vqgan_ckpt} does not exist.'
    assert os.path.isfile(args.vqgan_ckpt), f'{args.vqgan_ckpt} is not a file.'

    model = NewsgenBase(hparams)
    model.init_tokenizer(args.vqgan_ckpt, vqgan_hparams)

    # Load trainer
    trainer = Trainer(max_epochs=args.epochs,
                      accelerator=args.accelerator,
                      strategy=args.strategy,
                      devices='auto',
                      val_check_interval=args.val_every_n_steps,
                      log_every_n_steps=args.log_every_n_steps,
                      logger=logger,
                      callbacks=[ckpt_callback])

    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # Save model
    trainer.save_checkpoint(os.path.join(args.ckpt_dir, 'last.ckpt'))

    # Delete the trainer to free up memory
    del trainer

    # Test model
    trainer = Trainer(accelerator=args.accelerator,
                      devices=1,
                      num_nodes=1,
                      logger=logger)

    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
