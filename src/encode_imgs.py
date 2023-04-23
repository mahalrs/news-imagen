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
from data import VQVisualNewsDataset
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from vqgan.model import VQModel
from data import VQVisualNewsDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='data/visual_news_mini',
                    help='Directory containing VisualNews dataset')
parser.add_argument('--save_dataset',
                    default='data/vqgan_encodings',
                    help='Directory name to use when saving encodings')


with open('../src/hparams_vqgan.json', 'r') as f:
        hparams = json.load(f)
model = VQModel.load_from_checkpoint('../src/pretrained/vqgan.ckpt', **hparams)
model.init_lpips_from_pretrained('../src/pretrained/vgg.pth')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def encode(dataset_dir, save_dataset_dir):
    for split in ['train', 'val', 'test']:
        image_set = VQVisualNewsDataset(dataset_dir, split, transform)
        with open(os.path.join(dataset_dir, 'headlines.json'), 'r') as f:
            i = 0
            for story in json.load(f)[split]:
                image_path = os.path.join(dataset_dir, story['image_path'][2:])
                img = Image.open(image_path).convert('RGB')
                transformed = transform(img)
                img_tensor = transform(img).unsqueeze(0)
                output, loss, info = model.encode(img_tensor)
                # torch.save(output, os.path.join(save_dataset_dir, f"{story['id']}.pt"))
                
                np_encoding = output.detach().numpy()
                np.save(os.path.join(save_dataset_dir, f"{story['id']}.npy"), np_encoding)
                
                #outputs real & decoded images every 20 images for comparision
                i += 1
                if i % 20 == 1:
                    img_tensor = img_tensor / 2 + 0.5 #unnormalize
                    transforms.ToPILImage()(img_tensor.squeeze(0)).save(
                        os.path.join(save_dataset_dir, f"{story['id']}_real.jpg"))
                    decode_test(save_dataset_dir, story['id'])
                    print(i)
                    
def decode_test(save_dataset_dir, story_id):
    encoding = np.load(os.path.join(save_dataset_dir, f"{story_id}.npy"))
    encoding = torch.from_numpy(encoding)
    rec = model.decode(encoding).detach()
    rec = rec / 2 + 0.5 #unnormalize
    image = transforms.ToPILImage()(rec.squeeze(0))
    image.save(os.path.join(save_dataset_dir, f"{story_id}.jpg"))

def main():
    args = parser.parse_args()
    dataset_dir = args.dataset
    save_dataset_dir = args.save_dataset

    torch.set_grad_enabled(False)
    model.eval()
    
    encode(dataset_dir, save_dataset_dir)
    # decode_test(save_dataset_dir, 1466837)

if __name__ == '__main__':
    main()