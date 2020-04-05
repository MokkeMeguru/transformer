import torch.nn.functional as F
from tqdm import tqdm
from typing import List
from omegaconf import DictConfig
import hydra
import numpy as np
import torch
import random
from models.layers.utils import util as model_util
from metrics import bleu, earlystoping
from models.transformer import Transformer
from datasets.dataset import get_dataloader, BOS, PAD, EOS
from optimizers.normopt import NormOpt
from metrics.bleu import BLEUScore
from metrics.earlystoping import EarlyStopping
from datasets.vocab import PAD
from torch import optim
from torchtext import data
import torch.nn as nn
import logging
from logging import getLogger
from tensorboardX import SummaryWriter
logger = getLogger(__name__)


class Task:
    def __init__(self, hparams, device: torch.device):
        self.device = device
        self.basic_params = hparams["basic"]
        self.encoder_params = hparams["encoder"]
        self.decoder_params = hparams["decoder"]

        self.load_dataset()
        self.load_model()
        self.model.to(self.device)
        self.optimizer = NormOpt(
            d_model=self.basic_params["transformer"]["d_model"],
            factor=2.0,
            warmup_step=self.basic_params["warmup_step"],
            optimizer=optim.Adam(self.model.parameters(),
                                 lr=1e-5))

        self.loss = nn.CrossEntropyLoss(
            ignore_index=PAD,
            size_average=False).to(self.device)

        self.bleu = BLEUScore(EOS)
        self.early_stopping = EarlyStopping()

        self.model.to(self.device)
        self.writer = SummaryWriter(logdir="./summary")

    def load_dataset(self):
        datasets, vocabs = get_dataloader(
            self.basic_params["paths"]["train_base"],
            self.basic_params["paths"]["val_base"],
            self.basic_params["paths"]["test_base"],
            self.basic_params["paths"]["ext"])

        self.src_vocabs, self.tgt_vocabs = vocabs
        self.basic_params["vocab"]["num_src_vocab"] = len(self.src_vocabs.id2word)
        self.basic_params["vocab"]["num_tgt_vocab"] = len(self.tgt_vocabs.id2word)
        logger.info("source vocab size: {}".format(self.basic_params["vocab"]["num_src_vocab"]))
        logger.info("target vocab size: {}".format(self.basic_params["vocab"]["num_tgt_vocab"]))
        self.train_dataloader = data.BucketIterator(
            datasets[0],
            batch_size=self.basic_params["train"]["batch_size"],
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))

        self.val_dataloder = data.BucketIterator(
            datasets[1],
            batch_size=self.basic_params["valid"]["batch_size"],
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))

        self.test_dataloader = data.Iterator(
            datasets[2],
            batch_size=1)

    def load_model(self):
        self.model = Transformer(
            basic_params=self.basic_params,
            encoder_params=self.encoder_params,
            decoder_params=self.decoder_params,
            src_pad_idx=PAD,
            tgt_pad_idx=PAD)
        if self.basic_params["paths"]["prev_model"]:
            self.model.load_state_dict(
                torch.load(self.basic_params["paths"]["prev_model"]))
            ckpt = self.model.state_dict()
            torch.save(ckpt, self.basic_params["ckpt_path"])

    def train_step(self,
                   batch_X: torch.Tensor,
                   batch_Y: torch.Tensor):
        batch_X = batch_X.to(self.device)
        batch_Y = batch_Y.to(self.device)
        src_mask = model_util.padding_mask(
            batch_X, pad_id=PAD).to(self.device)
        tgt_mask = model_util.look_ahead_mask(
            batch_Y[:, :-1], pad_id=PAD).to(self.device)


        pred_Y = self.model(batch_X, batch_Y[:, :-1], src_mask, tgt_mask)
        gold = batch_Y[:, 1:].contiguous()
        loss = self.loss(pred_Y.view(-1, pred_Y.size(-1)), gold.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # gold = gold.data.cpu().numpy().tolist()
        # pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().tolist()
        return loss.item() # , gold, pred

    def val_step(self,
                   batch_X: torch.Tensor,
                   batch_Y: torch.Tensor):
        batch_X = batch_X.to(self.device)
        batch_Y = batch_Y.to(self.device)
        src_mask = model_util.padding_mask(
            batch_X, pad_id=PAD).to(self.device)
        tgt_mask = model_util.look_ahead_mask(
            batch_Y[:, :-1],  pad_id=PAD).to(self.device)
        pred_Y = self.model(batch_X, batch_Y[:, :-1], src_mask, tgt_mask)
        gold = batch_Y[:, 1:].contiguous()
        loss = self.loss(pred_Y.view(-1, pred_Y.size(-1)), gold.view(-1))
        gold = gold.data.cpu().numpy().tolist()
        pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().tolist()
        return loss.item(), gold, pred

    def train(self, trial=None):
        train_loss = 0.
        val_loss = 0.
        for epoch in range(1, self.basic_params["train"]["num_epochs"] + 1):
            if self.early_stopping.early_stop:
                break

            train_loss = 0.
            val_loss = 0.
            val_refs = []
            val_hyps = []

            self.model.train()
            if trial is None:
                for batch in tqdm(self.train_dataloader):
                    loss = self.train_step(batch.src, batch.trg)
                    train_loss += loss
            else:
                for batch in self.train_dataloader:
                    loss = self.train_step(batch.src, batch.trg)
                    train_loss += loss
            self.model.eval()
            with torch.no_grad():
                for batch in self.val_dataloder:
                    loss, gold, pred = self.val_step(batch.src, batch.trg)
                    val_loss += loss
                    val_refs += gold
                    val_hyps += pred
                train_loss /= len(self.train_dataloader.dataset)
                val_loss /= len(self.val_dataloder.dataset)
                val_bleu = self.bleu.calc_bleu(val_refs, val_hyps)

                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("valid/loss", val_loss, epoch)
                self.writer.add_scalar("valid/bleu", val_bleu, epoch)
                if trial is not None:
                    trial.report(val_loss, epoch)
                    if trial.should_prune(epoch):
                        import optuna
                        raise optuna.exceptions.TrialPruned()

                if self.early_stopping(val_loss):
                    ckpt = self.model.state_dict()
                    torch.save(ckpt, self.basic_params["ckpt_path"])
                logger.info(
                    "Epoch {}: train_loss: {:5.2f} valid_loss: {:5.2f} valid_bleu: {:2.2f}"\
                    .format(
                        epoch, train_loss, val_loss, val_bleu
                    ))
        return val_loss

    def greedy_decode(self, enc_output, src_mask):
        batch_size = enc_output.size(0)
        tgt_seq = torch.full([batch_size, 1], BOS,
                             dtype=torch.long, device=self.device)
        for i in range(1, self.basic_params["tgt_max_seq_len"]):
            tgt = self.model.pos_encoding(self.model.tgt_embed(tgt_seq))
            tgt_mask = model_util.look_ahead_mask(tgt_seq)
            dec_output, dec_slf_attns, dec_enc_attns = self.model.decoder(
                tgt, enc_output, tgt_mask, src_mask
            )
            output = F.log_softmax(self.model.generator(dec_output), dim=-1)
            output = output[:, -1, :].max(dim=-1)[1].unsqueeze(1)
            tgt_seq = torch.cat([tgt_seq, output], dim=-1)
        return tgt_seq[:, 1:], dec_slf_attns, dec_enc_attns

    def eval(self):
        self.model.load_state_dict(torch.load(self.basic_params["ckpt_path"]))
        self.model.eval()
        pred_Y = []
        with torch.no_grad():
            with open(self.basic_params["paths"]["test_base"] + ".en", "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    src = [self.src_vocabs.encode(line.strip().split())]
                    src = torch.tensor(src, dtype=torch.long)
                    src = src.to(self.device)
                    batch_size = src.size(0)
                    assert batch_size == 1, "not supported batch_size > 1"
                    src_mask = model_util.padding_mask(
                        src, pad_id=PAD).to(self.device)
                    src = self.model.pos_encoding(self.model.src_embed(src))
                    enc_output, enc_slf_attns = self.model.encoder(src, src_mask)
                    pred_y = self.greedy_decode(
                        enc_output, src_mask)[0].data.cpu().numpy().tolist()
                    preds = [pred[:pred.index(EOS)]
                             if EOS in pred else pred for pred in pred_y]
                    preds = [self.tgt_vocabs.decode(pred)for pred in preds]
                    pred_Y += preds

        with open('submission.csv', 'w', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f, delimiter=' ', lineterminator='\n')
            writer.writerows(pred_Y)


@hydra.main(config_path='./conf/config.yaml')
def main(cfg: DictConfig):
    from hydra import utils
    cfg.basic.paths.train_base = utils.to_absolute_path(cfg.basic.paths.train_base)
    cfg.basic.paths.val_base = utils.to_absolute_path(cfg.basic.paths.val_base)
    cfg.basic.paths.test_base = utils.to_absolute_path(cfg.basic.paths.test_base)
    if cfg.basic.paths.prev_model:
        cfg.basic.paths.prev_model = utils.to_absolute_path(cfg.basic.paths.prev_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsk = Task(hparams=cfg, device=device)
    tsk.train()
    tsk.eval()
    tsk.writer.export_scalars_to_json("./summary/all_scalars.json")
    tsk.writer.close()

if __name__ == '__main__':
    main()
