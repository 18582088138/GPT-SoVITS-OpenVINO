from pathlib import Path
import openvino as ov
import numpy as np
import torch

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    hann_window = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

class OVGptSoVits:
    def __init__(self, voice_name, device="CPU", EOS=1024, early_stop_num=2700, filter_length=2048, sampling_rate=32000, hop_length=640, win_length=2048):
        self.core = ov.Core()
        self.encoder_model = self.core.compile_model(f"IR_model/{voice_name}/t2s_encoder.xml", device)
        self.first_stage_decoder = self.core.compile_model(f"IR_model/{voice_name}/t2s_fsdec.xml", device)
        self.stage_decoder = self.core.compile_model(f"IR_model/{voice_name}/t2s_sdec.xml", device)

        self.vits_model = self.core.compile_model(f"IR_model/{voice_name}/vits.xml", device)

        self.EOS = EOS
        self.early_stop_num = early_stop_num
        self.filter_length = filter_length
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length

    def generate_refer(self, ref_audio):
        refer = spectrogram_torch(
            ref_audio,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False
        )

        return refer
    
    def __call__(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        refer = self.generate_refer(ref_audio)
        audio = self.vits_model([text_seq, pred_semantic, refer])[0]
        return audio

    def t2s(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.early_stop_num

        res = self.encoder_model([ref_seq, text_seq, ref_bert, text_bert, ssl_content])
        x, prompts = res[0], res[1]

        prefix_len = prompts.shape[1]

        #[1,N,512] [1,N]
        res = self.first_stage_decoder([x, prompts])

        y = res[0]
        k = res[1]
        v = res[2]
        y_emb = res[3]
        x_example = res[4]

        stop = False
        for idx in range(1, 1500):
            #[1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            enco = self.stage_decoder([y, k, v, y_emb, x_example])
            y, k, v, y_emb, logits, samples = enco[0], enco[1], enco[2], enco[3], enco[4], enco[5]
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if np.argmax(logits, axis=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return np.expand_dims(y[:, -idx:], 0)