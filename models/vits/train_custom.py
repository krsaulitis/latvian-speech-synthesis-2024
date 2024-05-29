import argparse
import os
import subprocess
import time

import pandas as pd
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import commons
import utils
from data_custom import CustomDatasetV1
from data_utils import (
    TextAudioSpeakerCollate,
)
from logger import Logger
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from text.symbols import symbols
from text.symbols_lv import symbols as symbols_lv

global_step = 0


def run():
    global global_step

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    hps = utils.get_hparams()
    logger = Logger(hps)
    torch.manual_seed(hps.train.seed)

    train_loader, test_loader = load_datasets(hps)
    models, epoch_start, step = load_models(hps, device)
    mod_gen, mod_dis, opt_gen, opt_dis = models

    for batch, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, text_raw) in tqdm(enumerate(test_loader),
                                                                                            leave=False):
        for idx, y_i in enumerate(y):
            print(text_raw[idx])
            if idx >= 3:
                break  # do not generate more than three samples per batch

    exit()

    global_step = int(step)
    epoch_start = int(epoch_start)
    print(f'Global step: {global_step}, Epoch start: {epoch_start}')

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_gen, gamma=hps.train.lr_decay, last_epoch=epoch_start - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_dis, gamma=hps.train.lr_decay, last_epoch=epoch_start - 2)

    for epoch in range(epoch_start, hps.train.epochs + 1):
        print(f'Epoch {epoch}/{hps.train.epochs}, global step {global_step}')

        for batch, data in tqdm(enumerate(train_loader)):
            if device == 'cuda' and batch == 0:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                cached = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f'Batch: {batch}, Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB')

            train(epoch, batch, hps, models, data, logger, device)

            if global_step % hps.train.log_interval == 0 and global_step > 0:
                logger.log_results(epoch, global_step)

            global_step += 1

        # Evaluate
        evaluate(epoch, hps, models, test_loader, logger, device)

        utils.save_checkpoint(mod_gen, opt_gen, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, f"G_{global_step}.pth"))
        utils.save_checkpoint(mod_dis, opt_dis, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, f"D_{global_step}.pth"))

        # Update learning rate
        scheduler_g.step()
        scheduler_d.step()


def train(epoch, batch, hps, models, data, logger, device):
    (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, text_raw) = data
    (mod_gen, mod_dis, opt_gen, opt_dis) = models

    mod_gen.train()
    mod_dis.train()

    x, x_lengths = x.to(device), x_lengths.to(device)
    spec, spec_lengths = spec.to(device), spec_lengths.to(device)
    y, y_lengths = y.to(device), y_lengths.to(device)
    if hps.data.n_speakers > 1:
        speakers = speakers.to(device)
    else:
        speakers = None

    y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q) = mod_gen(x, x_lengths, spec, spec_lengths, speakers)

    mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax)
    y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
    y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )

    y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

    # Discriminator
    y_d_hat_r, y_d_hat_g, _, _ = mod_dis(y, y_hat.detach())

    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    loss_disc_all = loss_disc

    opt_dis.zero_grad()
    loss_disc_all.backward()
    grad_norm_d = commons.clip_grad_value_(mod_dis.parameters(), None)
    opt_dis.step()

    # Generator loss
    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = mod_dis(y, y_hat)
    loss_dur = torch.sum(l_length.float())
    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

    loss_fm = feature_loss(fmap_r, fmap_g) * hps.train.c_fm
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

    opt_gen.zero_grad()
    loss_gen_all.backward()
    grad_norm_g = commons.clip_grad_value_(mod_gen.parameters(), None)
    opt_gen.step()

    logger.step_train(
        gen_learning_rate=opt_gen.param_groups[0]['lr'],
        gen_grad_norm=grad_norm_g,
        dis_grad_norm=grad_norm_d,
        duration_loss=loss_dur.detach().cpu().item(),
        mel_loss=loss_mel.detach().cpu().item(),
        kl_loss=loss_kl.detach().cpu().item(),
        feature_loss=loss_fm.detach().cpu().item(),
        gen_loss=loss_gen.detach().cpu().item(),
        gen_total_loss=loss_gen_all.detach().cpu().item(),
        dis_loss=loss_disc_all.detach().cpu().item(),
    )


def evaluate(epoch, hps, models, test_loader, logger, device):
    with torch.no_grad():
        (mod_gen, mod_dis, opt_gen, opt_dis) = models

        mod_gen.eval()
        mod_dis.eval()

        eval_idx = 0

        start_time_test = time.time()
        print("Testing...")
        for batch, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, text_raw) in tqdm(enumerate(test_loader),
                                                                                                leave=False):
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            speakers = speakers.to(device)
            if hps.data.n_speakers > 1:
                speakers = speakers.to(device)
            else:
                speakers = None

            y_hat, attn, mask, *_ = mod_gen.infer(x, x_lengths, speakers, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            y_hat_sliced, ids_slice = commons.rand_slice_segments(y_hat, y_hat_lengths, hps.train.segment_size, y_lengths)
            y_sliced = commons.slice_segments(y, ids_slice, hps.train.segment_size)

            y_mel = mel_spectrogram_torch(
                y_sliced.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat_sliced.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = mod_dis(y_sliced, y_hat_sliced)

            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_fm = feature_loss(fmap_r, fmap_g) * hps.train.c_fm
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_dis, losses_dis_r, losses_dis_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

            logger.step_test(
                mel_loss=loss_mel.detach().cpu().item(),
                feature_loss=loss_fm.detach().cpu().item(),
                gen_loss=loss_gen.detach().cpu().item(),
                dis_loss=loss_dis.detach().cpu().item(),
            )

            for idx, y_i in enumerate(y_hat):
                eval_idx += 1
                image_dict = utils.plot_spectrogram_to_numpy(y_hat_mel[idx].cpu().numpy())
                audio_dict = y_i[:, :y_hat_lengths[idx]]
                attn_dict = utils.plot_alignment_to_numpy(attn[idx].cpu().numpy())

                if not os.path.exists(f'./{hps.train.tmp_dir}'):
                    os.makedirs(f'./{hps.train.tmp_dir}')

                sf.write(f'./{hps.train.tmp_dir}/{eval_idx}.wav', audio_dict.squeeze().cpu().numpy(), hps.data.sampling_rate)

                logger.add_image(image_dict, eval_idx, text_raw[idx])
                logger.add_attention(attn_dict, eval_idx, text_raw[idx])
                logger.add_audio(audio_dict.squeeze().cpu().numpy(), eval_idx, text_raw[idx])
                if idx >= 3:
                    break  # do not generate more than three samples per batch

        print(f'Testing time: {time.time() - start_time_test:.2f}s')

        # Calculate NISQA
        start_time_nisqa = time.time()
        subprocess.run(
            ['python', '../nisqa/run_predict.py',
             '--mode', 'predict_dir',
             '--pretrained_model', '../nisqa/weights/nisqa.tar',
             '--data_dir', f'./{hps.train.tmp_dir}',
             '--num_workers', '4',
             '--bs', '16',
             '--output_dir', f'./{hps.train.tmp_dir}']
        )
        if os.path.exists(f'./{hps.train.tmp_dir}/NISQA_results.csv'):
            data = pd.read_csv(f'./{hps.train.tmp_dir}/NISQA_results.csv')
            avg_quality = data['mos_pred'].mean()

            logger.step_test(avg_quality=avg_quality)

        os.system(f'rm -rf ./{hps.train.tmp_dir}')
        print(f'NISQA time: {time.time() - start_time_nisqa:.2f}s')


def load_datasets(hps):
    print('Loading datasets...')
    collate_fn = TextAudioSpeakerCollate()

    train_dataset = CustomDatasetV1(hps.data.training_files, hps.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_dataset = CustomDatasetV1(hps.data.validation_files, hps.data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=hps.train.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader


def load_models(hps, device):
    print('Loading models...')
    symbols_length = len(symbols_lv) if hps.data.use_phonemes else len(symbols)

    mod_gen = SynthesizerTrn(
        symbols_length,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)

    mod_dis = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)

    opt_gen = torch.optim.AdamW(
        mod_gen.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    opt_dis = torch.optim.AdamW(
        mod_dis.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    try:
        print(f'Loading checkpoint {hps.model_dir}...')
        mod_gen, opt_gen, learning_rate, epoch_start, step = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, 'G_*.pth'), mod_gen, opt_gen)
        mod_dis, opt_dis, learning_rate, epoch_start, step = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, 'D_*.pth'), mod_dis, opt_dis)
        print(f'Loaded checkpoint from epoch {epoch_start} at step {step}...')
    except Exception as e:
        print(e)
        print('No checkpoint found, starting from scratch...')
        step = 0
        epoch_start = 1

    return (mod_gen, mod_dis, opt_gen, opt_dis), epoch_start, step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run()
