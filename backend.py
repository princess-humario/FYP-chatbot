import json
import os
from typing import List, Dict, Tuple
from pathlib import Path
import openai
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

class MultimodalLLMWrapper:
   def __init__(self, api_key: str):
       self.api_key = api_key
       openai.api_key = api_key
       os.environ["OPENAI_API_KEY"] = api_key
       self.llm = OpenAI(temperature=0)
       self.client = openai.OpenAI(api_key=api_key)
   
   def describe_image(self, base64_image: str) -> str:
       try:
           response = self.client.chat.completions.create(
               model="o3-pro", #gpt-4-vision-preview deprecated model, not good, change according to your need but if you need
               #to process scientific diagrams I would suggest 03-pro but i would suggest you take a look at the following
               #link: https://platform.openai.com/docs/models 
               messages=[{
                   "role": "user",
                   "content": [
                       {
                           "type": "text", 
                           "text": "Describe this scientific diagram/image in detail for document search. Focus on key elements, data, and findings shown."
                       },
                       {
                           "type": "image_url", 
                           "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                       }
                   ]
               }],
               max_tokens=300
           )
           return response.choices[0].message.content
       except Exception as e:
           return f"[Image description unavailable: {str(e)}]"
   
   def analyze_displacement_image(self, base64_image: str, context_question: str = "") -> str:
       try:
           prompt = f"""
           Analyze this image in the context of displacement sensing research. 
           {context_question}
           
           Please identify and describe:
           1. Any displacement measurement equipment or sensors visible
           2. Laser interferometry setups or optical components
           3. Mechanical measurement systems
           4. Experimental configurations
           5. Data, graphs, or measurement results shown
           6. Technical specifications or parameters visible
           
           Provide a detailed technical analysis suitable for research purposes.
           """
           
           response = self.client.chat.completions.create(
               model="chatgpt-4o-latest", 
               messages=[{
                   "role": "user",
                   "content": [
                       {"type": "text", "text": prompt},
                       {
                           "type": "image_url", 
                           "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                       }
                   ]
               }],
               max_tokens=1000
           )
           return response.choices[0].message.content
       except Exception as e:
           return f"Error analyzing displacement image: {str(e)}"
   
   def generate_technical_diagram(self, prompt: str, diagram_type: str = "schematic", size: str = "1024x1024") -> str:
       try:
           enhanced_prompt = f"""
           Technical scientific diagram: {prompt}
           
           Style: {diagram_type}
           Requirements:
           - Scientific accuracy and precision
           - Clear labeling of components
           - Professional engineering diagram style
           - Suitable for research paper inclusion
           - Clean, high-contrast design
           - Detailed technical illustration
           
           Focus on displacement sensing, measurement systems, optical setups, and scientific equipment.
           """
           
           response = self.client.images.generate(
               model="gpt-image-1", #since here more focus is being done on the power signal so i have used the latest model as of yet 14/06/2025
               prompt=enhanced_prompt,
               size=size,
               quality="hd",
               style="natural",
               n=1
           )
           
           return response.data[0].url
           
       except Exception as e:
           return f"Error generating diagram: {str(e)}"
   
   def detect_image_generation_request(self, message: str) -> bool:
       generation_keywords = [
           "generate", "create", "make", "draw", "design", "diagram", 
           "show me", "illustrate", "sketch", "plot", "chart", "schematic"
       ]
       
       message_lower = message.lower()
       return any(keyword in message_lower for keyword in generation_keywords)

class PromptTemplates:
   @staticmethod
   def get_qa_template():
       template = """
       You are a scientific paper assistant. Based on the paper content provided, answer the question accurately.
       
       Paper Content: {context}
       
       Question: {question}
       
       Instructions:
       - If the information exists in the paper, provide a detailed answer
       - If the information is NOT in the paper, respond with: "I couldn't find information about that topic in this paper. The paper may not cover this specific aspect."
       - Always base your answer only on the provided content
       
       Answer:
       """
       return PromptTemplate(template=template, input_variables=["context", "question"])

class PaperIndexer:
   def __init__(self, llm_wrapper: MultimodalLLMWrapper):
       self.llm_wrapper = llm_wrapper
       self.embeddings = OpenAIEmbeddings()
       self.text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=1000,
           chunk_overlap=200
       )
       self.vectorstores = {}
   
   def load_paper_from_json(self, file_path: str, paper_name: str) -> List[Document]:
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               data = json.load(f)
           
           documents = []
           
           if isinstance(data, dict) and "content" in data:
               for item in data["content"]:
                   content = self._process_json_item(item, item.get("page", 0))
                   if content:
                       doc = Document(
                           page_content=content,
                           metadata={'source': paper_name, 'page': item.get("page", 0)}
                       )
                       documents.append(doc)
           elif isinstance(data, list):
               for i, item in enumerate(data):
                   content = self._process_json_item(item, i)
                   if content:
                       doc = Document(
                           page_content=content,
                           metadata={'source': paper_name, 'page': i}
                       )
                       documents.append(doc)
           
           return documents
           
       except Exception as e:
           raise Exception(f"Error loading {paper_name}: {str(e)}")
   
   def load_papers_from_folder(self, folder_path: str) -> Dict[str, bool]:
       results = {}
       folder = Path(folder_path)
       
       if not folder.exists():
           print(f"Folder '{folder_path}' does not exist!")
           return results
       
       json_files = list(folder.glob("*.json"))
       
       if not json_files:
           print(f"No JSON files found in '{folder_path}'")
           return results
       
       print(f"Found {len(json_files)} JSON files to load...")
       
       for json_file in json_files:
           paper_name = json_file.stem
           try:
               print(f"Loading: {paper_name}")
               self.create_paper_index(str(json_file), paper_name)
               results[paper_name] = True
               print(f"✓ Successfully loaded: {paper_name}")
           except Exception as e:
               print(f"✗ Failed to load {paper_name}: {str(e)}")
               results[paper_name] = False
       
       return results
   
   def _process_json_item(self, item: Dict, page_num: int) -> str:
       text_content = item.get('text', '') or item.get('content', '')
       
       images = item.get('images', [])
       image_descriptions = []
       
       if images:
           for img_base64 in images:
               description = self.llm_wrapper.describe_image(img_base64)
               image_descriptions.append(f"[Image Description: {description}]")
       
       full_content = text_content
       if image_descriptions:
           full_content += "\n\n" + "\n".join(image_descriptions)
       
       return full_content
   
   def create_paper_index(self, file_path: str, paper_name: str) -> Chroma:
       if paper_name in self.vectorstores:
           return self.vectorstores[paper_name]
       
       documents = self.load_paper_from_json(file_path, paper_name)
       doc_chunks = self.text_splitter.split_documents(documents)
       
       vectorstore = Chroma.from_documents(
           documents=doc_chunks,
           embedding=self.embeddings,
           collection_name=paper_name.replace(" ", "_").lower()
       )
       
       self.vectorstores[paper_name] = vectorstore
       return vectorstore
   
   def get_cached_papers(self) -> List[str]:
       return list(self.vectorstores.keys())
   
   def clear_cache(self):
       self.vectorstores = {}

class PaperQA:
   def __init__(self, api_key: str):
       self.llm_wrapper = MultimodalLLMWrapper(api_key)
       self.indexer = PaperIndexer(self.llm_wrapper)
       self.prompt_template = PromptTemplates.get_qa_template()
   
   def load_paper(self, file_path: str, paper_name: str) -> bool:
       try:
           self.indexer.create_paper_index(file_path, paper_name)
           return True
       except Exception as e:
           print(f"Error loading paper: {e}")
           return False
   
   def load_papers_from_folder(self, folder_path: str) -> Dict[str, bool]:
       return self.indexer.load_papers_from_folder(folder_path)
   
   def query_paper(self, paper_name: str, question: str) -> Tuple[str, List[str]]:
       try:
           if paper_name not in self.indexer.vectorstores:
               return "Paper not loaded. Please load the paper first.", []
           
           vectorstore = self.indexer.vectorstores[paper_name]
           
           qa_chain = RetrievalQA.from_chain_type(
               llm=self.llm_wrapper.llm,
               chain_type="stuff",
               retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
               return_source_documents=True,
               chain_type_kwargs={"prompt": self.prompt_template}
           )
           
           response = qa_chain({"query": question})
           sources = [doc.page_content[:300] + "..." for doc in response['source_documents']]
           
           return response['result'], sources
           
       except Exception as e:
           return f"Error querying paper: {str(e)}", []
   
   def analyze_image(self, base64_image: str, context_question: str = "") -> str:
       return self.llm_wrapper.analyze_displacement_image(base64_image, context_question)

   def generate_diagram(self, prompt: str, diagram_type: str = "schematic", size: str = "1024x1024") -> str:
       return self.llm_wrapper.generate_technical_diagram(prompt, diagram_type, size)
   
   def get_available_papers(self) -> List[str]:
       return self.indexer.get_cached_papers()
   
   def clear_all_papers(self):
       self.indexer.clear_cache()

if __name__ == "__main__":
   api_key = os.getenv("OPENAI_API_KEY")
   if api_key:
       qa_system = PaperQA(api_key)
       folder_path = "json_papers"
       results = qa_system.load_papers_from_folder(folder_path)
       print(f"\nLoaded papers: {qa_system.get_available_papers()}")
   else:
       print("API key not found in environment variables")
