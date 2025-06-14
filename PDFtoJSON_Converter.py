import json
import os
import base64
from typing import List, Dict, Any
import fitz
from PIL import Image
import io
from pathlib import Path

class PDFToJSON:
   def __init__(self, input_folder: str, output_folder: str):
       self.input_folder = Path(input_folder)
       self.output_folder = Path(output_folder)
       self.output_folder.mkdir(parents=True, exist_ok=True)
       self.supported_formats = {'.pdf', '.txt'}
   
   def convert_all_papers(self) -> Dict[str, bool]:
       results = {}
       
       if not self.input_folder.exists():
           print(f"Input folder '{self.input_folder}' does not exist!")
           return results
       
       paper_files = []
       for ext in self.supported_formats:
           paper_files.extend(list(self.input_folder.glob(f"*{ext}")))
       
       if not paper_files:
           print(f"No supported files found in '{self.input_folder}'")
           return results
       
       print(f"Found {len(paper_files)} papers to convert...")
       
       for paper_file in paper_files:
           try:
               print(f"Converting: {paper_file.name}")
               success = self._convert_single_paper(paper_file)
               results[paper_file.name] = success
               
               if success:
                   print(f"✓ Successfully converted: {paper_file.name}")
               else:
                   print(f"✗ Failed to convert: {paper_file.name}")
                   
           except Exception as e:
               print(f"✗ Error converting {paper_file.name}: {str(e)}")
               results[paper_file.name] = False
       
       return results
   
   def _convert_single_paper(self, file_path: Path) -> bool:
       try:
           if file_path.suffix.lower() == '.pdf':
               paper_data = self._process_pdf(file_path)
           elif file_path.suffix.lower() == '.txt':
               paper_data = self._process_txt(file_path)
           else:
               return False
           
           output_file = self.output_folder / f"{file_path.stem}.json"
           with open(output_file, 'w', encoding='utf-8') as f:
               json.dump(paper_data, f, indent=2, ensure_ascii=False)
           
           return True
           
       except Exception as e:
           print(f"Error processing {file_path.name}: {str(e)}")
           return False
   
   def _process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
       paper_data = []
       
       doc = fitz.open(pdf_path)
       
       for page_num in range(len(doc)):
           page = doc[page_num]
           text = page.get_text()
           
           images = []
           image_list = page.get_images()
           
           for img_index, img in enumerate(image_list):
               try:
                   xref = img[0]
                   pix = fitz.Pixmap(doc, xref)
                   
                   if pix.n - pix.alpha < 4:
                       img_data = pix.tobytes("png")
                       img_base64 = base64.b64encode(img_data).decode('utf-8')
                       images.append(img_base64)
                   
                   pix = None
                   
               except Exception as e:
                   print(f"Warning: Could not extract image {img_index} from page {page_num + 1}: {str(e)}")
           
           page_data = {
               "page": page_num + 1,
               "text": text.strip(),
               "images": images,
               "metadata": {
                   "source_file": pdf_path.name,
                   "page_number": page_num + 1,
                   "total_pages": len(doc)
               }
           }
           
           paper_data.append(page_data)
       
       doc.close()
       return paper_data
   
   def _process_txt(self, txt_path: Path) -> List[Dict[str, Any]]:
       with open(txt_path, 'r', encoding='utf-8') as f:
           content = f.read()
       
       chunks = self._split_text_into_chunks(content, chunk_size=2000)
       
       paper_data = []
       for i, chunk in enumerate(chunks):
           page_data = {
               "page": i + 1,
               "text": chunk.strip(),
               "images": [],
               "metadata": {
                   "source_file": txt_path.name,
                   "chunk_number": i + 1,
                   "total_chunks": len(chunks)
               }
           }
           paper_data.append(page_data)
       
       return paper_data
   
   def _split_text_into_chunks(self, text: str, chunk_size: int = 2000) -> List[str]:
       if len(text) <= chunk_size:
           return [text]
       
       chunks = []
       words = text.split()
       current_chunk = []
       current_length = 0
       
       for word in words:
           if current_length + len(word) + 1 <= chunk_size:
               current_chunk.append(word)
               current_length += len(word) + 1
           else:
               if current_chunk:
                   chunks.append(' '.join(current_chunk))
               current_chunk = [word]
               current_length = len(word)
       
       if current_chunk:
           chunks.append(' '.join(current_chunk))
       
       return chunks
   
   def get_conversion_summary(self, results: Dict[str, bool]) -> None:
       successful = sum(results.values())
       total = len(results)
       failed = total - successful
       
       print("\n" + "="*50)
       print("CONVERSION SUMMARY")
       print("="*50)
       print(f"Total papers: {total}")
       print(f"Successfully converted: {successful}")
       print(f"Failed: {failed}")
       
       if failed > 0:
           print("\nFailed files:")
           for filename, success in results.items():
               if not success:
                   print(f"  ✗ {filename}")
       
       print(f"\nJSON files saved to: {self.output_folder}")
       print("="*50)

def main():
   INPUT_FOLDER = "input_papers"
   OUTPUT_FOLDER = "json_papers"
   
   print("Scientific Paper to JSON Converter")
   print("="*40)
   
   converter = PDFToJSON(INPUT_FOLDER, OUTPUT_FOLDER)
   results = converter.convert_all_papers()
   converter.get_conversion_summary(results)

if __name__ == "__main__":
   try:
       import fitz
       from PIL import Image
   except ImportError:
       print("Required packages not found. Please install them using:")
       print("pip install PyMuPDF Pillow")
       exit(1)
   
   main()
