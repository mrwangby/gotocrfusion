import os
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, as_completed

'''def process_pdf(pdf_path):
    """处理单个 PDF 文件，将其转换为图像并保存。"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.dirname(pdf_path)  # 使用 PDF 文件的父目录作为输出文件夹
    output_folder = os.path.join(output_folder, pdf_name)  # 在父目录下创建以 PDF 名称命名的文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        print(f"Converting {pdf_path} to images in {output_folder}...")
        images = convert_from_path(pdf_path)
        
        for i, image in enumerate(images):
            image_filename = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.jpg")
            image.save(image_filename, "JPEG")
            print(f"Saved image: {image_filename}")
        
        return output_folder  # 返回处理完成的文件夹路径
    except Exception as e:
        print(f"Skipping {pdf_path} due to error: {e}")
        return None

def pdf_to_images_in_folders(root_dir, max_workers=2):
    """将目录中的所有 PDF 转换为图像，并返回每个 PDF 的图像文件夹路径。"""
    pdf_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    print(f"Found {len(pdf_paths)} PDF files.")

    pdf_folders = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in pdf_paths}
        
        for future in as_completed(future_to_pdf):
            result = future.result()
            if result:
                pdf_folders.append(result)

    print("All PDF files have been converted.")
    return pdf_folders'''

import os
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_pdf(pdf_path):
    """处理单个 PDF 文件，将其转换为图像并保存，并返回生成的图像数量。"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.dirname(pdf_path)  # 使用 PDF 文件的父目录作为输出文件夹
    output_folder = os.path.join(output_folder, pdf_name)  # 在父目录下创建以 PDF 名称命名的文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    image_count = 0
    try:
        print(f"Converting {pdf_path} to images in {output_folder}...")
        images = convert_from_path(pdf_path)
        
        for i, image in enumerate(images):
            image_filename = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.jpg")
            image.save(image_filename, "JPEG")
            print(f"Saved image: {image_filename}")
            image_count += 1  # 统计图像数量
        
        return output_folder, image_count  # 返回处理完成的文件夹路径和生成的图像数量
    except Exception as e:
        print(f"Skipping {pdf_path} due to error: {e}")
        return None, 0

def pdf_to_images_in_folders(root_dir, max_workers=2):
    """将目录中的所有 PDF 转换为图像，并返回每个 PDF 的图像文件夹路径以及图像总数量。"""
    pdf_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    print(f"Found {len(pdf_paths)} PDF files.")

    pdf_folders = []
    total_images = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in pdf_paths}
        
        for future in as_completed(future_to_pdf):
            folder, image_count = future.result()
            if folder:
                pdf_folders.append(folder)
                total_images += image_count  # 累加每个 PDF 的图像数量

    print("All PDF files have been converted.")
    print(f"Total PDF documents: {len(pdf_paths)}")
    print(f"Total images generated: {total_images}")
    return pdf_folders, len(pdf_paths), total_images



# 使用时指定根目录路径
pdf_to_images_in_folders("your_directory_path")
