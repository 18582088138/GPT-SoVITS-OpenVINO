import requests
import zipfile
import shutil
import os

def download_pretrain_model(voice_name = "Nagisa", output_path='./pretrained_models/'):
    os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
    print("== Check environment variable == ",os.getenv("HF_ENDPOINT"))

    #@title Import model 导入模型 (HuggingFace)
    hf_link = 'https://hf-mirror.com/modelloosrvcc/Nagisa_Shingetsu_GPT-SoVITS/resolve/main/Nagisa.zip' #@param {type: "string"}
    source_directory = output_path
    SoVITS_destination_directory = f'./{source_directory}/{voice_name}/SoVITS_weights'
    GPT_destination_directory = f'./{source_directory}/{voice_name}/GPT_weights'

    if not os.path.exists(source_directory):
        os.makedirs(source_directory)
    if not os.path.exists(SoVITS_destination_directory):
        os.makedirs(SoVITS_destination_directory)
    if not os.path.exists(GPT_destination_directory):
        os.makedirs(GPT_destination_directory)
    sovits_destination_path = os.path.join(SoVITS_destination_directory, f"{voice_name}_SOVITS.pth")
    gpt_destination_path = os.path.join(GPT_destination_directory, f"{voice_name}_GPT.ckpt")

    downloaded_file = os.path.join(output_path, 'file.zip')
    if not os.path.exists(downloaded_file):
        response = requests.get(hf_link)
        with open(downloaded_file, 'wb') as file:
            file.write(response.content)

        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print("==GPT-SoVITs pretrain model download success==")

        # os.remove(f"{output_path}/file.zip")

    for filename in os.listdir(source_directory):
        if filename.endswith(".pth"):
            source_path = os.path.join(source_directory, filename)
            shutil.move(source_path, sovits_destination_path)

    for filename in os.listdir(source_directory):
        if filename.endswith(".ckpt"):
            source_path = os.path.join(source_directory, filename)
            shutil.move(source_path, gpt_destination_path)

    print(f'Model downloaded. ',sovits_destination_path, gpt_destination_path)
    return sovits_destination_path, gpt_destination_path

download_pretrain_model(voice_name = "Nagisa", output_path='./pretrained_models')