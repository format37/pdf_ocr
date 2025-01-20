import fitz  # PyMuPDF
import io
import os
import PIL.Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from pathlib import Path

class PDFImageExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.output_dir = Path('extracted_images')
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_with_pymupdf(self):
        """
        Extract images using PyMuPDF (fitz) library.
        This method is good for extracting embedded images.
        """
        print("Extracting images with PyMuPDF...")
        doc = fitz.open(self.pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image extension
                ext = base_image["ext"]
                
                # Save the image
                image_path = self.output_dir / f'pymupdf_page{page_num + 1}_img{img_idx + 1}.{ext}'
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_bytes)
                    
        doc.close()

    def extract_with_pdf2image(self):
        """
        Convert PDF pages to images using pdf2image.
        This method is good for capturing the entire page as an image.
        """
        print("Extracting images with pdf2image...")
        pages = convert_from_path(self.pdf_path)
        
        for idx, page in enumerate(pages):
            image_path = self.output_dir / f'pdf2image_page{idx + 1}.png'
            page.save(str(image_path), 'PNG')
            
            # Apply image processing to detect and extract sub-images
            self._process_page_image(str(image_path), idx + 1)

    def _process_page_image(self, image_path, page_num):
        """
        Process a page image to detect and extract potential sub-images using
        computer vision techniques.
        """
        # Read the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and extract potential image regions
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w < 50 or h < 50:
                continue
                
            # Extract the region
            region = img[y:y+h, x:x+w]
            
            # Save the region
            region_path = self.output_dir / f'cv_page{page_num}_region{idx + 1}.png'
            cv2.imwrite(str(region_path), region)

    def extract_with_ocr(self):
        """
        Use OCR to detect text regions and potentially identify diagrams/figures.
        """
        print("Extracting images with OCR...")
        pages = convert_from_path(self.pdf_path)
        
        for idx, page in enumerate(pages):
            # Convert PIL Image to numpy array
            img_np = np.array(page)
            
            # Get OCR data
            ocr_data = pytesseract.image_to_data(img_np, output_type=pytesseract.Output.DICT)
            
            # Find regions without text (potential images)
            height, width = img_np.shape[:2]
            mask = np.ones((height, width), dtype=np.uint8) * 255
            
            # Mark text regions in mask
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 0:  # Filter confident text detections
                    x, y, w, h = (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['width'][i],
                        ocr_data['height'][i]
                    )
                    mask[y:y+h, x:x+w] = 0
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract potential image regions
            for cont_idx, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small regions
                if w < 100 or h < 100:
                    continue
                
                region = img_np[y:y+h, x:x+w]
                region_path = self.output_dir / f'ocr_page{idx + 1}_region{cont_idx + 1}.png'
                cv2.imwrite(str(region_path), cv2.cvtColor(region, cv2.COLOR_RGB2BGR))

    def extract_all(self):
        """
        Run all extraction methods.
        """
        self.extract_with_pymupdf()
        self.extract_with_pdf2image()
        self.extract_with_ocr()
        
        # Remove duplicate images
        self._remove_duplicates()

    def _remove_duplicates(self):
        """
        Remove duplicate images based on image hash comparison.
        """
        image_files = list(self.output_dir.glob('*'))
        hashes = {}
        
        for image_path in image_files:
            try:
                with PIL.Image.open(image_path) as img:
                    # Convert to grayscale and resize for hash comparison
                    img_small = img.convert('L').resize((32, 32))
                    img_array = np.array(img_small)
                    img_hash = hash(img_array.tobytes())
                    
                    if img_hash in hashes:
                        # If duplicate found, remove the current file
                        image_path.unlink()
                    else:
                        hashes[img_hash] = image_path
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

def main():
    # Example usage
    pdf_path = "assets/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    extractor = PDFImageExtractor(pdf_path)
    extractor.extract_all()

if __name__ == "__main__":
    main()