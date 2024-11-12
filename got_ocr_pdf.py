import os
import multiprocessing

import fitz
import torch
from transformers import AutoTokenizer
import run_ocr_crop


def pdf_to_images(pdf_path: str, output_folder: str) -> list[str]:
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    pdf_document.close()

    return image_paths


def load_model(model_path: str) -> tuple:
    run_ocr_crop.disable_torch_init()
    model_name = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = run_ocr_crop.GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    model.to(device='cuda',  dtype=torch.bfloat16)

    return model, tokenizer


def process_image(model, tokenizer, img_path: str) -> str:
    image_processor_high =  run_ocr_crop.BlipImageEvalProcessor(image_size=1024)
    use_im_start_end = True
    image_token_len = 256

    # qs = 'OCR with format upon the patch reference: '
    qs = 'OCR with format upon the patch reference. Please express in markdown format. If there is a formula, please use LaTeX format (wrapped in the symbol $) to express the formula: '
    
    image_list = []
    img = run_ocr_crop.load_image(img_path)
    sub_images = run_ocr_crop.dynamic_preprocess(img)
    ll = len(sub_images)

    for p in sub_images:
        image = p
        image_1 = image.copy()
        image_tensor_1 = image_processor_high(image_1)
        image_list.append(image_tensor_1)

    image_list = torch.stack(image_list)

    # qs = args.query
    if use_im_start_end:
        qs = run_ocr_crop.DEFAULT_IM_START_TOKEN + run_ocr_crop.DEFAULT_IMAGE_PATCH_TOKEN*image_token_len*ll + run_ocr_crop.DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = run_ocr_crop.DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"

    conv = run_ocr_crop.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != run_ocr_crop.SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = run_ocr_crop.KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = run_ocr_crop.TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_list.half().cuda(), image_list.half().cuda())],
            do_sample=False,
            num_beams = 1,
            # no_repeat_ngram_size = 20,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
        )
        
    # print('==============rendering===============')
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    right_num = outputs.count('\\right')
    left_num = outputs.count('\left')

    if right_num != left_num:
        outputs = outputs.replace('\left(', '(').replace('\\right)', ')').replace('\left[', '[').replace('\\right]', ']').replace('\left{', '{').replace('\\right}', '}').replace('\left|', '|').replace('\\right|', '|').replace('\left.', '.').replace('\\right.', '.')

    outputs = outputs.replace('"', '``').replace('$', '')

    outputs_list = outputs.split('\n')
    gt= ''
    for out in outputs_list:
        gt +=  '"' + out.replace('\\', '\\\\') + r'\n' + '"' + '+' + '\n' 
    
    gt = gt[:-2]

    return gt


def process_and_save(img_path):
    response = process_image(model, tokenizer, img_path)
    result_file_path = os.path.join(pdf_output_folder, f"{os.path.basename(img_path)}.txt")
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        result_file.write(response)


if __name__ == "__main__":
    input_folder = "./pdf_test2"
    output_folder = "./outputs"
    os.makedirs(output_folder, exist_ok=True)

    model_path = "stepfun-ai/GOT-OCR2_0"
    model, tokenizer = load_model(model_path)

    # 遍历PDF文件夹中的所有PDF文件
    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, pdf_file)
            pdf_output_folder = os.path.join(output_folder, os.path.splitext(pdf_file)[0])
            os.makedirs(pdf_output_folder, exist_ok=True)

            # 将PDF转换为图像
            image_paths = pdf_to_images(pdf_path, pdf_output_folder)

            # # 对每个图像进行OCR处理
            # for img_path in image_paths:
            #     response = process_image(model, tokenizer, img_path)
            #     # 保存结果
            #     result_file_path = os.path.join(pdf_output_folder, f"{os.path.basename(img_path)}.txt")
            #     with open(result_file_path, 'w', encoding='utf-8') as result_file:
            #         result_file.write(response)

            # # 对每个图像进行OCR处理
            result_file_path = os.path.join(pdf_output_folder, f"{os.path.basename(pdf_file)}.txt")
            with open(result_file_path, 'w', encoding='utf-8') as result_file:
                for img_path in image_paths:
                    response = process_image(model, tokenizer, img_path)
                    result_file.write(response)

            # pool = multiprocessing.Pool(processes=4)
            # pool.map(process_and_save, image_paths)
            # pool.close()
            # pool.join()

            print(f"Processed {pdf_file} and saved results to {pdf_output_folder}")

