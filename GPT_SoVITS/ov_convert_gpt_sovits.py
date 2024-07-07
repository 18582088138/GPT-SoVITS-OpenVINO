
import torch
import torchaudio
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

from text import cleaned_text_to_sequence
import soundfile
import os
import json
import openvino as ov
from ov_gpt_sovits import OVGptSoVits
from feature_extractor import cnhubert
from module.models_onnx import SynthesizerTrn, symbols
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule

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


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class T2SEncoder(nn.Module):
    def __init__(self, t2s, vits):
        super().__init__()
        self.encoder = t2s.onnx_encoder
        self.vits = vits
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        # bert = ref_bert.transpose(0, 1)
        bert = torch.cat([ref_bert.transpose(0, 1), text_bert.transpose(0, 1)], 1)
        all_phoneme_ids = torch.cat([ref_seq, text_seq], 1)
        bert = bert.unsqueeze(0)
        prompt = prompt_semantic.unsqueeze(0)
        return self.encoder(all_phoneme_ids, bert), prompt


class T2SModel(nn.Module):
    def __init__(self, t2s_path, vits_model):
        super().__init__()
        dict_s1 = torch.load(t2s_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "ojbk", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        self.t2s_model.eval()
        self.vits_model = vits_model.vq_model
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model.model.top_k = torch.LongTensor([self.config["inference"]["top_k"]])
        self.t2s_model.model.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.t2s_model = self.t2s_model.model
        self.t2s_model.init_onnx()
        self.onnx_encoder = T2SEncoder(self.t2s_model, self.vits_model)
        self.first_stage_decoder = self.t2s_model.first_stage_decoder
        self.stage_decoder = self.t2s_model.stage_decoder
        #self.t2s_model = torch.jit.script(self.t2s_model)

    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content):
        early_stop_num = self.t2s_model.early_stop_num
        print(f"EARLY STOP NUM {early_stop_num}")
        print(f"EOS {self.t2s_model.EOS}")

        #[1,N] [1,N] [N, 1024] [N, 1024] [1, 768, N]
        x, prompts = self.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)

        prefix_len = prompts.shape[1]

        #[1,N,512] [1,N]
        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        stop = False
        for idx in range(1, 1500):
            #[1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
            enco = self.stage_decoder(y, k, v, y_emb, x_example)
            y, k, v, y_emb, logits, samples = enco
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.t2s_model.EOS or samples[0, 0] == self.t2s_model.EOS:
                stop = True
            if stop:
                break
        y[0, -1] = 0

        return y[:, -idx:].unsqueeze(0)

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name, dynamo=False):

        ov_model = ov.convert_model(
            self.onnx_encoder,
            example_input=(ref_seq, text_seq, ref_bert, text_bert, ssl_content),
        )
        ov.save_model(ov_model, f"IR_model/{project_name}/t2s_encoder.xml")
        print("== Export t2s_encoder IR model sucess ==")
        x, prompts = self.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)

        #print(self.first_stage_decoder)
        ov_model = ov.convert_model(
            self.first_stage_decoder,
            example_input=(x, prompts.to(torch.int32)),
        )
        ov.save_model(ov_model, f"IR_model/{project_name}/t2s_fsdec.xml")
        print("== Export t2s_fsdec IR model sucess ==")
        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        ov_model = ov.convert_model(
            self.stage_decoder,
            example_input=(y.to(torch.int32), k, v, y_emb, x_example),
        )
        ov.save_model(ov_model, f"IR_model/{project_name}/t2s_sdec.xml",)
        print("== Export t2s_sdec IR model sucess ==")


class VitsModel(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        dict_s2 = torch.load(vits_path,map_location="cpu")
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

    def generate_refer(self, ref_audio):
        print("==== ref_audio ====",ref_audio.size())
        print(f"==== FILTER LENGTH {self.hps.data.filter_length}")
        print(f"==== SAMPLING RATE {self.hps.data.sampling_rate}")
        print(f"==== HOP LENGTH {self.hps.data.hop_length,}")
        print(f"==== WIN LENGTH {self.hps.data.win_length,}")
        refer = spectrogram_torch(
            ref_audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        )
        return refer
        
    def forward(self, text_seq, pred_semantic, refer):
        return self.vq_model(pred_semantic, text_seq, refer)[0, 0]


class GptSoVits(nn.Module):
    def __init__(self, vits, t2s):
        super().__init__()
        self.vits = vits
        self.t2s = t2s
    
    def forward(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content, debug=False):
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        refer = self.vits.generate_refer(ref_audio)
        audio = self.vits(text_seq, pred_semantic, refer)
        return audio

    def export(self, ref_seq, text_seq, ref_bert, text_bert, ref_audio, ssl_content, project_name):
        self.t2s.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, project_name)
        pred_semantic = self.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        refer = self.vits.generate_refer(ref_audio)
        ov_model = ov.convert_model(self.vits, example_input=(text_seq, pred_semantic, refer))
        ov.save_model(ov_model, f"IR_model/{project_name}/vits.xml")

class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        cnhubert_base_path = "pretrained_models/chinese-hubert-base"
        cnhubert.cnhubert_base_path=cnhubert_base_path
        ssl_model = cnhubert.get_model()
        self.ssl = ssl_model

    def forward(self, ref_audio_16k):
        return self.ssl.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)

def export(vits_path, gpt_path, project_name):
    vits = VitsModel(vits_path)
    gpt = T2SModel(gpt_path, vits)
    gpt_sovits = GptSoVits(vits, gpt)
    ssl = SSLModel()
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(["n", "i2", "h", "ao3", ",", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"])])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(["w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"])])
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()
    ref_audio = torch.randn((1, 48000 * 5)).float()
    # ref_audio = torch.tensor([load_audio("rec.wav", 48000)]).float()
    ref_audio_16k = torchaudio.functional.resample(ref_audio,48000,16000).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio,48000,vits.hps.data.sampling_rate).float()

    try:
        os.makedirs(f"IR_model/{project_name}")
    except:
        pass

    ssl_content = ssl(ref_audio_16k).float()
    ov_model = ov.convert_model(ssl, example_input=ref_audio_16k)
    ov.save_model(ov_model, f"IR_model/{project_name}/ssl.xml")
     
    gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content, project_name)
    print("== Export gpt_sovits IR model sucess ==")

    MoeVSConf = {
            "Folder" : f"{project_name}",
            "Name" : f"{project_name}",
            "Type" : "GPT-SoVits",
            "Rate" : vits.hps.data.sampling_rate,
            "NumLayers": gpt.t2s_model.num_layers,
            "EmbeddingDim": gpt.t2s_model.embedding_dim,
            "Dict": "BasicDict",
            "BertPath": "chinese-roberta-wwm-ext-large",
            "Symbol": symbols,
            "AddBlank": False
        }
    
    MoeVSConfJson = json.dumps(MoeVSConf)
    with open(f"IR_model/{project_name}.json", 'w') as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent = 4)
    
    # Debug = True
    # if Debug:
    #     a = gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content).detach().cpu().numpy()
    #     soundfile.write("out.wav", a, vits.hps.data.sampling_rate)
    #     ov_gpt_sovits = OVGptSoVits(f"{project_name}")
    #     b = ov_gpt_sovits(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content.detach().cpu().numpy())
    #     soundfile.write("ov_out.wav", b, vits.hps.data.sampling_rate)
 
def export_bert(voice_name):
    bert_path = os.environ.get(
        "bert_path", "pretrained_models/chinese-roberta-wwm-ext-large"
    )
    text = "OpenVINO is the best AI inference toolkit."
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    bert_model = bert_model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    print("== inputs ==",inputs)
    ov_input = {"input_ids":inputs["input_ids"],
                "attention_mask":inputs["attention_mask"],
                "token_type_ids":inputs["token_type_ids"]}


    ov_bert_model = ov.convert_model(bert_model, example_input=ov_input)
    ov.save_model(ov_bert_model, f"./IR_model/{voice_name}/chinese-roberta.xml",compress_to_fp16=True)
    print("== Export Bert IR model sucess ==")
    pass


if __name__ == "__main__":
    try:
        os.mkdir("IR_model")
    except:
        pass
    
    voice_name = "Nagisa"
    sovits_path = f"./pretrained_models/{voice_name}/SoVITS_weights/{voice_name}_SOVITS.pth"
    gpt_path = f"./pretrained_models/{voice_name}/GPT_weights/{voice_name}_GPT.ckpt"

    export_bert(voice_name)
    export(sovits_path, gpt_path, voice_name)
    