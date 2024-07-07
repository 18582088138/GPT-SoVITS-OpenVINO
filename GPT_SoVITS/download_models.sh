echo "Download pretrain model"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download lj1995/GPT-SoVITS --local-dir pretrained_models
python download_gpt_sovits_model.py


