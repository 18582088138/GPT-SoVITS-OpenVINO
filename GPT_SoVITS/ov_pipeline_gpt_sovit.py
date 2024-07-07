import os, re, logging
import LangSegment
import pdb
import torch
import soundfile
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
import openvino as ov


from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
# from tools.i18n.i18n import I18nAuto

# i18n = I18nAuto()

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

device = "cpu"
is_half = "False"
gpt_path = os.environ.get("gpt_path", "./pretrained_models/Nagisa/GPT_weights/Nagisa_GPT.ckpt")
sovits_path = os.environ.get("sovits_path", "./pretrained_models/Nagisa/SoVITS_weights/Nagisa_SOVITS.pth")
bert_path = os.environ.get("bert_path", "./pretrained_models/chinese-roberta-wwm-ext-large")

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
bert_model = bert_model.to(device)

voice_name = "Nagisa" 
core = ov.Core()
ov_ssl_models_path = f"IR_model/{voice_name}/ssl.xml"
ov_ssl_model = core.compile_model(ov_ssl_models_path, "CPU")

# ov_bert_model_path = f"IR_model/{voice_name}/chinese-roberta.xml"
# ov_bert_model = core.compile_model(ov_bert_model_path, "CPU")

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        # ov_input = {"input_ids":inputs["input_ids"],
        #             "attention_mask":inputs["attention_mask"],
        #             "token_type_ids":inputs["token_type_ids"]}
        # res = ov_bert_model(ov_input)
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

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


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)


change_gpt_weights(gpt_path)

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

# dict_language = {
#     i18n("中文"): "all_zh",#全部按中文识别
#     i18n("英文"): "en",#全部按英文识别#######不变
#     i18n("日文"): "all_ja",#全部按日文识别
#     i18n("中英混合"): "zh",#按中英混合识别####不变
#     i18n("日英混合"): "ja",#按日英混合识别####不变
#     i18n("多语种混合"): "auto",#多语种启动切分识别语种
# }

def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text,language):
    if language in {"en","all_zh","all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones,bert.to(dtype),norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

from ov_gpt_sovits import OVGptSoVits
def get_tts_wav(voice_name, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="no_cut", top_k=20, top_p=0.6, temperature=0.6, ref_free = False):
    ov_gpt_sovits = OVGptSoVits(voice_name)

    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    # prompt_language = dict_language[prompt_language]
    # text_language = dict_language[text_language]
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text    
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            raise OSError("The reference audio is outside the range of 3 to 10 seconds, please change it!")
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        wav16k = wav16k.to(device)
        zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_start = ttime()
        ssl_content = ov_ssl_model(wav16k.unsqueeze(0))[0]
        # ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        ssl_end = ttime() 
        print("== ov ssl_content inference ==",ssl_end - ssl_start)

    t1 = ttime()

    # if (how_to_cut == i18n("凑四句一切")):
    #     text = cut1(text)
    # elif (how_to_cut == i18n("凑50字一切")):
    #     text = cut2(text)
    # elif (how_to_cut == i18n("按中文句号。切")):
    #     text = cut3(text)
    # elif (how_to_cut == i18n("按英文句号.切")):
    #     text = cut4(text)
    # elif (how_to_cut == i18n("按标点符号切")):
    #     text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        ref_text_start = ttime()
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language)
        ref_text_end = ttime()
        print("== torch ref_text time ==",ref_text_end - ref_text_start)

        refer_audio_start = ttime()
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        refer_audio_end = ttime()
        print("== torch refer_audio time ==",refer_audio_end - refer_audio_start)

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        # print(i18n("实际输入的目标文本(每句):"), text)
        target_text_start = ttime()
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language)
        target_text_end = ttime()
        print("== torch target_text time ==",target_text_end - target_text_start)
        # print(i18n("前端处理后的文本(每句):"), norm_text2)
        # bert = bert.to(device).unsqueeze(0)
        t2 = ttime()
        
        ref_seq = torch.LongTensor(phones1).unsqueeze(0)
        ref_bert = bert1.transpose(0,1)
        text_seq = torch.LongTensor(phones2).unsqueeze(0)
        text_bert = bert2.transpose(0,1)
        t2s_start = ttime()
        pred_semantic = ov_gpt_sovits.t2s(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
        t2s_end = ttime()
        print("== OpenVINO t2s time ==",t2s_end - t2s_start)
        pred_semantic = torch.from_numpy(pred_semantic)
        # print("==OpenVINO pred_semantic==",pred_semantic.shape, type(pred_semantic))
        t3 = ttime()
        
        # print("==get_spepc==",refer.shape,hps, ref_wav_path)
        refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        
        audio_gen_start = ttime()
        audio = ov_gpt_sovits.vits_model([text_seq, pred_semantic, refer])[0]
        audio_gen_end = ttime()
        print("== OpenVINO audio_gen time ==",audio_gen_end - audio_gen_start)

        # print("==OpenVINO audio==",audio.shape, type(audio))
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    t5 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    print("== Total time cost %.2f s =="%(t5-t0))
    return audio_opt

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts



if __name__ == "__main__":
    ref_audio_path = "datasets/GaoYe_Audio.WAV_0000000000_0000143360.wav"
    ref_text= "老师来了进吧，不好意思啊，高老师家里出了点事儿。"
    ref_language = "zh"
    # text = "OpenVINO 是最好的推理工具。支持CPU和GPU。支持Windows，Linux，MacOS，支持Python和C++等语言。支持各种深度学习框架，如TensorFlow，PyTorch，ONNX，Caffe等。"
    # text = "家里出了点事儿。我家司机知道这件事的细节。你们可以去问他"
    # text = r"请打开路径C:\Users\kundaxu\OneDriveIntel-Corporation\Desktop\IoTG\Customer"
    text = "你好，我是你的AI助手小惠，很高兴认识你。"
    text_language = "zh"
    voice_name = "Nagisa"
    sampling_rate = 16000
    time_start = ttime()
    audio_opt = get_tts_wav(voice_name, ref_audio_path, ref_text, ref_language, 
                text, text_language, how_to_cut="no_cut", 
                top_k=5, top_p=1, temperature=1, ref_free = False)
    audio_result = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    time_end = "%.2f"%(ttime() - time_start)
    soundfile.write(f"ov_{voice_name}_out_{time_end}s.wav", audio_result, 32768)
    print("==Finish==")