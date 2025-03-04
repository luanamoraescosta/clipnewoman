from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor
import os
import gc
from PyPDF2 import PdfReader  # Adding the import for PyPDF2

def pdf_to_images(pdf_path, output_folder, dpi=150):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # Read the total number of pages
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)

        # Process each page
        for page_num in range(1, total_pages + 1):
            images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
            if not images:
                break  # If there are no images, exit the loop
            for image in images:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_path = os.path.join(output_folder, f'{pdf_name}_page_{page_num}.png')
                image.save(output_path, 'PNG')
                print(f'Page {page_num} of PDF {pdf_name} saved as {output_path}')
                del image  # Free memory of the image after saving
                gc.collect()  # Trigger garbage collection after each page

    except Exception as e:
        print(f'Error processing {pdf_path}: {e}')

def convert_pdfs_in_folder(pdf_folder, output_folder, dpi=150):
    with ProcessPoolExecutor(max_workers=2) as executor:  # Limit the number of processes
        for root, dirs, files in os.walk(pdf_folder):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    relative_subfolder = os.path.relpath(root, pdf_folder)
                    output_folder_path = os.path.join(output_folder, relative_subfolder, os.path.splitext(file)[0])
                    executor.submit(pdf_to_images, pdf_path, output_folder_path, dpi)

if __name__ == '__main__':
    pdf_folder = "C:\\Users\\lumor\\Downloads\\my_pdfs_1911"
    output_folder = "E:\\The Delineator Images\\1911"
    convert_pdfs_in_folder(pdf_folder, output_folder)
