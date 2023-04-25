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
parser.add_argument('--dataset',
                    default='./data/visual_news',
                    help='Directory containing VisualNews dataset')
parser.add_argument('--encoded_data_file',
                    default='encoded_captions.json',
                    help='Name of file with encoded data')
parser.add_argument('--log_dir',
                    default='./logs',
                    help='Directory to save logs')
parser.add_argument('--ckpt_dir',
                    default='./checkpoints/newsgen',
                    help='Directory to save checkpoints')
parser.add_argument('--hparams',
                    default='./newsgen/hparams_base.json',
                    help='Path to hparams json file')
parser.add_argument('--vqgan_ckpt',
                    default='./pretrained/vqgan.ckpt',
                    help='Path to VQGAN checkpoint')
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
                    default=12,
                    type=int,
                    help='Number of epochs to train')
parser.add_argument('--val_every_n_steps',
                    default=10000,
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
    assert os.path.exists(args.dataset), f'{args.dataset} does not exist.'
    assert os.path.isdir(args.dataset), f'{args.dataset} is not a directory.'

    train_set = EncodedVisualNewsDataset(args.dataset, args.encoded_data_file,
                                         'train')
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch,
                              shuffle=(not args.distributed),
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_set = EncodedVisualNewsDataset(args.dataset, args.encoded_data_file,
                                       'val')
    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch,
                            shuffle=(not args.distributed),
                            num_workers=args.num_workers,
                            pin_memory=True)

    test_set = EncodedVisualNewsDataset(args.dataset, args.encoded_data_file,
                                        'test')
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch,
                             shuffle=(not args.distributed),
                             num_workers=args.num_workers,
                             pin_memory=True)

    # Define logger and checkpoint callback
    logger = TensorBoardLogger(args.log_dir, name='newsgen')

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='{epoch:02d}-{step:06d}',
        save_last=True,
        save_top_k=-1)

    # lr_scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    hparams['warmup_steps'] = warmup_steps
    hparams['training_steps'] = total_steps
    print(f'Warmup steps: {warmup_steps}')
    print(f'Total steps: {total_steps}')

    # Load model
    assert os.path.exists(args.vqgan_ckpt), f'{args.vqgan_ckpt} does not exist.'
    assert os.path.isfile(args.vqgan_ckpt), f'{args.vqgan_ckpt} is not a file.'

    model = NewsgenBase(hparams)
    model.init_tokenizer(args.vqgan_ckpt)

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

    # Test model
    trainer = Trainer(accelerator=args.accelerator,
                      devices=1,
                      num_nodes=1,
                      logger=logger)

    trainer.test(model, dataloaders=test_loader)

    # Save model
    trainer.save_checkpoint(os.path.join(args.ckpt_dir, 'last.ckpt'))


if __name__ == '__main__':
    main()
