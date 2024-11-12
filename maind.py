import os
import json
from datetime import datetime
from argparse import ArgumentParser
import time
import logging
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, CLIPImageProcessor
import torch.distributed as dist
import torch.multiprocessing as mp

import run_ocr
from pdf2png.pdf2png import pdf_to_images_in_folders
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from natsort import natsorted
from transformers import TextStreamer
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html
import string
import glob

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

outputFilePath = './allResult.json'
 
translation_table = str.maketrans(punctuation_dict)

def log_with_rank(rank, device, message):
    logging.info(f'Rank:{rank}, Device:{device} - {message}')


logging.basicConfig(
    filename='main1.log',  # 日志文件名
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

def init_processes(rank, world_size, fn, args,pdf_folders):
    os.environ['MASTER_ADDR'] = 'localhost'  # 或者你的主机地址
    os.environ['MASTER_PORT'] = '47423'
    device = f"cuda:{rank // 2}"  # 根据 rank 计算要使用的 GPU
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    fn(rank, world_size, args,pdf_folders,device)
    dist.destroy_process_group()

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--pdf_folder", type=str, default="./pdfs")
    parser.add_argument("--image_folder", type=str, default="./pdfs")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--models", nargs='+', 
                        default=["./model/model_weight"]
    )
    # parser.add_argument("--prompt", type=str, default="""
    #     识别图片中的文字内容，并给出markdown格式的输出，如果有公式, 公式请用latex格式输出。
    # """)
    parser.add_argument("--prompt", type=str, default="""
        what is the text in the image? Answer this question using the text in the image directly.
    """)
    return parser.parse_args()

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        return json.load(f)

def load_model(model_path: str, config: dict,device) -> tuple:
    if config['model_type'] == 'got':
        run_ocr.disable_torch_init()
        model_name = os.path.expanduser(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = run_ocr.GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
        model.to(device=device, dtype=torch.bfloat16)
        image_processor = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer, image_processor




def process_image(model, tokenizer, image_processor, img_path: str, prompt: str, config: dict, device) -> str:
    
    if config['call_method'] == 'got_ocr_generation':
        image_processor = BlipImageEvalProcessor(image_size=1024)
        image_processor_high = BlipImageEvalProcessor(image_size=1024)

        use_im_start_end = True
        image_token_len = 256

        qs = 'OCR with format. If there is a formula, please use LaTeX format (wrapped in the symbol $) to express the formula: '

        image = load_image(img_path)
        w, h = image.size

        if use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv_mode = "mpt"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print(prompt)

        inputs = tokenizer([prompt])

        image_tensor = image_processor(image).to(device)
        image_tensor_1 = image_processor_high(image.copy()).to(device)

        input_ids = torch.as_tensor(inputs.input_ids).to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids,
                images=[(image_tensor.unsqueeze(0).half(), image_tensor_1.unsqueeze(0).half())],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        right_num = outputs.count('\\right')
        left_num = outputs.count('\\left')

        if right_num != left_num:
            outputs = outputs.replace('\\left(', '(').replace('\\right)', ')').replace('\\left[', '[').replace('\\right]', ']').replace('\\left{', '{').replace('\\right}', '}').replace('\\left|', '|').replace('\\right|', '|').replace('\\left.', '.').replace('\\right.', '.')

        outputs = outputs.replace('"', '``').replace('$', '')

        response = outputs
    
    return response


'''def save_pdf_results_to_json(name, text, output_folder):
    """将每个 PDF 的识别结果保存为单个 JSON 文件，包含 name 和 text 字段"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{name}_{timestamp}.json"
    json_path = os.path.join(output_folder, json_filename)

    pdf_result = {
        "name": f"{name}.pdf",
        "text": text
    }

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(pdf_result, json_file, ensure_ascii=False, indent=4)
    logging.info(f"Results for {name} saved in {json_filename}")'''
def save_pdf_results_to_json(name, text, output_folder, output_filename="001.json"):
    # 设置主JSON文件的路径
    json_path = os.path.join(output_folder, output_filename)
    
    # 创建新的结果条目
    pdf_result = {
        "name": f"{name}.pdf",
        "text": text,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # 检查JSON文件是否已存在
    if os.path.exists(json_path):
        # 如果文件存在，加载已有数据
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    else:
        # 如果文件不存在，则初始化为空列表
        data = []

    # 将新的结果条目添加到数据列表中
    data.append(pdf_result)

    # 将更新后的数据写回JSON文件
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    logging.info(f"已将{name}的结果保存到{json_path}")

'''def save_pdf_results_to_json(name, text, output_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = outputFilePath

    pdf_result = {
        "name": f"{name}.pdf",
        "text": text
    }

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(pdf_result, json_file, ensure_ascii=False, indent=4)
    logging.info(f"Results for {name} saved in {json_path}")
'''
def main(rank, world_size, args, pdf_folders,device):
    start_time = time.time()  # 记录程序开始时间
    #device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    total_config = load_config('config.json')
    os.makedirs(args.output_folder, exist_ok=True)

    # 根据 rank 和 world_size 划分 pdf_folders
    num_folders = len(pdf_folders)
    folders_per_process = num_folders // world_size
    start_index = rank * folders_per_process
    if rank == world_size - 1:  # 确保最后一个进程处理剩余文件夹
        assigned_folders = pdf_folders[start_index:]
    else:
        assigned_folders = pdf_folders[start_index:start_index + folders_per_process]

    for model_path in args.models:
        time_start = time.time()
        model_config = total_config[model_path]
        model, tokenizer, image_processor = load_model(model_path, model_config, device)
        model_name = model_path.split('/')[-1]
        log_with_rank(rank,device, f"Load model took: {time.time() - time_start:.2f} seconds")
        # 对每个 PDF 的图片文件夹进行 OCR，并将结果保存为单独的 JSON 文件
        for pdf_folder in assigned_folders:  # 使用划分后的文件夹
            pdf_name = os.path.basename(pdf_folder)
            combined_text = ""  # 用于存储每个 PDF 的所有页面识别结果
            
            for img_name in tqdm(os.listdir(pdf_folder), desc=f"Processing {pdf_name} with {model_name}"):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(pdf_folder, img_name)
                    imageprocess_time = time.time()
                    response = process_image(model, tokenizer, image_processor, img_path, args.prompt, model_config, device)
                    log_with_rank(rank, device,f"process ocr per image: {time.time() - imageprocess_time:.2f} seconds")
                    combined_text += response + "\n"

            save_pdf_results_to_json(pdf_name, combined_text.strip(), args.output_folder,outputfiles="results.json")

        time_end = time.time()
        log_with_rank(rank, device,f"Total process time for: {time_end - time_start:.2f} seconds")

    total_time = time.time() - start_time
    log_with_rank(rank,device, f"Total program runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    args = get_args()
    #world_size = torch.cuda.device_count()*2
    world_size=16
    print(world_size)
    # 1. 首先将 PDF 文件转换为图片，并获取每个 PDF 的图片文件夹路径
    allstart_time = time.time()
    pdf_folders,numpdf,numpng = pdf_to_images_in_folders(args.image_folder, max_workers=2)
    logging.info(f'Total PDF documents processed: {numpdf}')
    logging.info(f'Total images generated: {numpng}')
    logging.info(f'process image took {time.time() - allstart_time:.2f} seconds')
    
    # 确保 pdf_folders 的划分在这里进行
    mp.spawn(init_processes, args=(world_size, main, args, pdf_folders), nprocs=world_size)
    logging.info(f'all process spend time: {time.time() - allstart_time:.2f} seconds')
