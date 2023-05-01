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

import copy

import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchvision

from transformers import BartForConditionalGeneration, BartConfig, AdamW, get_linear_schedule_with_warmup
from .tokenizer import NewsgenTokenizer


class NewsgenBase(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        # Get the pretrained encoder from BART
        model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large')
        self.encoder = model.get_encoder()

        # Freeze the encoder if specified
        if hparams['freeze_encoder']:
            for layer in self.encoder.parameters():
                layer.requires_grad = False

        # New decoder with weights set to random values
        config = copy.deepcopy(model.config).to_dict()
        config.update(hparams['decoder_config'])
        config = BartConfig(**config)

        self.decoder_start_token_id = config.decoder_start_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id

        model = BartForConditionalGeneration(config=config)
        self.decoder = model.get_decoder()

        # New linear layer to project the decoder output to the vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.loss = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

        self.learning_rate = hparams['learning_rate']
        self.warmup_steps = hparams['warmup_steps']
        self.training_steps = hparams['training_steps']
        self.monitor = hparams['monitor']
        self.tokenizer = None

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def init_tokenizer(self, vqgan_ckpt_path, hparams):
        self.tokenizer = NewsgenTokenizer(vqgan_ckpt_path,
                                          hparams=hparams,
                                          device=self.device)

    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, decoder_input_ids, encoder_hidden_states,
               encoder_attention_mask):
        return self.decoder(input_ids=decoder_input_ids,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask)

    def forward(self, input_ids, attention_mask, labels):
        encoder_outputs = self.encode(input_ids, attention_mask)

        # Shift labels one token to the right
        decoder_input_ids = labels.new_zeros(labels.shape)
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()
        decoder_input_ids[:, 0] = self.decoder_start_token_id

        decoder_outputs = self.decode(decoder_input_ids, encoder_outputs[0],
                                      attention_mask)

        last_hidden_state = decoder_outputs[0]
        lm_logits = self.lm_head(last_hidden_state)
        # skip bias

        # Cross entropy loss ignores padding tokens
        loss = self.loss(lm_logits.view(-1, lm_logits.shape[-1]),
                         labels.view(-1))

        return loss, lm_logits

    def generate(self, input_ids, attention_mask):
        encoder_outputs = self.encode(input_ids, attention_mask)
        decoder_input_ids = torch.ones((input_ids.shape[0], 257),
                                       dtype=torch.int).to(self.device)
        decoder_input_ids *= self.pad_token_id
        decoder_input_ids[:, 0] = self.bos_token_id

        generated_tokens = torch.zeros((input_ids.shape[0], 256),
                                       dtype=torch.int).to(self.device)

        for i in range(1, 257):
            decoder_outputs = self.decode(decoder_input_ids, encoder_outputs[0],
                                          attention_mask)
            last_hidden_state = decoder_outputs[0]
            lm_logits = self.lm_head(last_hidden_state)

            next_token = self.tokenizer.get_indices(lm_logits)[:, i - 1]
            generated_tokens[:, i - 1] = next_token
            decoder_input_ids[:, i] = next_token

        return generated_tokens

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_eval_step(batch, batch_idx, 'train')

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        sch.step()

        # Log metrics
        self.log('train_loss',
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self._shared_eval_step(batch, batch_idx, 'val', True)

        # Log metrics
        self.log('val_loss',
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)

        return loss, logits

    def test_step(self, batch, batch_idx):
        loss, logits = self._shared_eval_step(batch, batch_idx, 'test', True)

        # Log metrics
        self.log('test_loss',
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)

        return loss, logits

    def _shared_eval_step(self, batch, batch_idx, phase, log_images=False):
        input_ids, input_mask, labels = batch

        # Run the model and get the loss and logits
        loss, logits = self(input_ids=input_ids,
                            attention_mask=input_mask,
                            labels=labels)

        if batch_idx % 100 == 0 and log_images:
            self.log_images(labels, logits, phase)

        return loss, logits

    def log_images(self, tgt_ids, logits, phase):
        tgt_rec = self.tokenizer.decode_images_code(tgt_ids[:4])
        pred_rec = self.tokenizer.decode_images(logits[:4])

        tgt_grid = torchvision.utils.make_grid(tgt_rec) / 2 + 0.5  # unnormalize
        pred_grid = torchvision.utils.make_grid(
            pred_rec) / 2 + 0.5  # unnormalize

        self.logger.experiment.add_image(f'{phase}_{self.global_step}_original',
                                         tgt_grid, self.global_step)

        self.logger.experiment.add_image(
            f'{phase}_{self.global_step}_predicted', pred_grid,
            self.global_step)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.training_steps)

        return [optimizer], [lr_scheduler]
