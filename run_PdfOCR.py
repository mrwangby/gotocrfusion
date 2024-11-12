import fitz  # PyMuPDF
import numpy as np
import os
import requests
import cv2
import base64
import multiprocessing
import time
import random

# 设置OpenAI API密钥
API_KEY = os.environ.get("OPENAI_API_KEY")
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

def enhance_image(image, scale_factor=2):
    # 放大图像
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    enlarged = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # 转换为灰度图
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 降噪
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def ask_gpt4o_vision(image, prompt, max_retries=4, delay=1):
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # 将图像编码为 base64 字符串
    _, buffer = cv2.imencode(".jpg", image)
    image = base64.b64encode(buffer).decode('utf-8')


    # 准备请求体
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2048
    }

    for attempt in range(max_retries):
        try:
            # 发送请求
            response = requests.post(API_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # 如果响应状态码不是 200，将引发异常
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']
            else:
                raise KeyError("No 'choices' in API response")
        except (requests.RequestException, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay + random.uniform(0, 2))  # 添加一些延迟
            else:
                raise Exception(f"Failed to get valid response after {max_retries} attempts: {str(e)}")


def process_page(args):
    prompt, page_number, pdf_path, output_path = args
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)

        pix = page.get_pixmap()
        img_path = os.path.join(output_path, f"page_{page_number + 1:03d}.png")
        pix.save(img_path)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # 图像预处理
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = enhance_image(image)
        text_content = ask_gpt4o_vision(image, prompt=prompt)
        
        md_path = os.path.join(output_path, f"page_{page_number + 1:03d}.md")
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(text_content["content"])
    except Exception as e:
        print(f"Error processing page {page_number + 1}: {str(e)}")
        # 可以选择在这里记录错误，或者重新抛出异常

def merge_md_files(folder_path, output_file="whole.md"):
    # 确保输出文件的扩展名是 .md
    if not output_file.endswith('.md'): output_file += '.md'
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 检查文件是否是 .md 文件
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    # 写入文件名作为标题
                    outfile.write(f"# {file}\n\n")
                    # 读取并写入文件内容
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    outfile.write("\n\n")  # 在文件之间添加空行

    print(f"合并完成，输出文件：{output_file}")



def process_pdf(pdf_path, output_path, prompt):
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    if not os.path.exists(output_path): os.makedirs(output_path)

    # 创建进程池
    pool = multiprocessing.Pool(processes=4)
    
    # 准备参数
    args = [(prompt, i, pdf_path, output_path) for i in range(num_pages)]
    
    # 并行处理页面
    pool.map(process_page, args)
    
    # 关闭进程池
    pool.close()
    pool.join()
        
if __name__ == "__main__":

    # prompt
    prompt = "你是核聚变领域的理论专家，精通该领域的全部现有知识，对专业名词以及数学推导尤其精通，\
              现在需要你帮我将教科书图片中的文字准确无误识别，但请注意以下4点： \
              1.务必保证全部文字的准确性，不要擅自篡改任何内容，尤其仔细核实所有数学符号和公式；\
              2.如果文中存在插图，请取出图片的名称，使用{页码}-{图号}-{图名}标识； \
              3.对于目录页，请整理为表格形式，对于非目录页，请务必删除所有页眉、页脚和页码，不要在输出中包含这些内容, \
              4.输出转换为markdown格式，但输出字符串中无需添加markdown的前缀。"
            #    如果文中存在插图，请提取出图片的名称，并截取相关区域截取为图像。"

    pdf_path = "example.pdf"
    output_path = "output"

    # 使用示例
    multiprocessing.freeze_support()
    process_pdf(pdf_path, output_path, prompt)

    output_path = "gpt-4o"
    merge_md_files(output_path)