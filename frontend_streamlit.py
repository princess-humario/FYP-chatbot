import streamlit as st
import os
from dotenv import load_dotenv
from t import PaperQA
import base64
from PIL import Image
import io
import time
from datetime import datetime

load_dotenv()

def process_image(uploaded_file):
   try:
       image = Image.open(uploaded_file)
       width, height = image.size
       format_type = image.format
       mode = image.mode
       
       buffered = io.BytesIO()
       image.save(buffered, format=format_type)
       img_base64 = base64.b64encode(buffered.getvalue()).decode()
       
       return {
           "image": image,
           "width": width,
           "height": height,
           "format": format_type,
           "mode": mode,
           "base64": img_base64,
           "size_mb": len(buffered.getvalue()) / (1024 * 1024)
       }
   except Exception as e:
       st.error(f"Error processing image: {str(e)}")
       return None

def detect_image_generation_request(message):
   generation_keywords = [
       "generate", "create", "make", "draw", "design", "diagram", 
       "show me", "illustrate", "sketch", "plot", "chart", "schematic"
   ]
   
   message_lower = message.lower()
   return any(keyword in message_lower for keyword in generation_keywords)

def query_all_papers(paper_qa, question):
   available_papers = paper_qa.get_available_papers()
   
   if not available_papers:
       return "No papers loaded to answer questions.", []
   
   all_answers = []
   all_sources = []
   
   for paper in available_papers:
       answer, sources = paper_qa.query_paper(paper, question)
       if "couldn't find information" not in answer.lower():
           all_answers.append(f"**From {paper}:**\n{answer}")
           all_sources.extend(sources)
   
   if all_answers:
       combined_answer = "\n\n".join(all_answers)
       return combined_answer, all_sources
   else:
       return "I couldn't find information about that topic in the available papers.", []

st.set_page_config(
   page_title="Information about displacement sensing", 
   layout="wide"
)

if 'paper_qa' not in st.session_state:
   api_key = os.getenv("OPENAI_API_KEY")
   if api_key:
       st.session_state.paper_qa = PaperQA(api_key)
       json_folder = "json_papers"
       if os.path.exists(json_folder):
           st.session_state.paper_qa.load_papers_from_folder(json_folder)
   else:
       st.session_state.paper_qa = None

if 'messages' not in st.session_state:
   st.session_state.messages = []

if 'current_image' not in st.session_state:
   st.session_state.current_image = None

st.title("Information about displacement sensing")
st.markdown("*Ask questions or generate diagrams using all available research papers*")

if st.session_state.paper_qa:
   available_papers = st.session_state.paper_qa.get_available_papers()
   
   col1, col2 = st.columns([3, 1])
   
   with col1:
       if available_papers:
           st.info(f"**Active Context:** Using all {len(available_papers)} research papers for comprehensive responses")
       else:
           st.warning("No papers loaded - responses will be limited")
       
       st.subheader("Upload Image (Optional)")
       uploaded_file = st.file_uploader(
           "Upload displacement sensing image to enhance responses:",
           type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
           help="Upload experimental setups, diagrams, or equipment for enhanced context"
       )
       
       if uploaded_file is not None:
           image_data = process_image(uploaded_file)
           
           if image_data:
               col_img, col_info = st.columns([1, 2])
               
               with col_img:
                   st.image(image_data["image"], caption="Uploaded Image", width=200)
               
               with col_info:
                   st.success("Image uploaded and ready!")
                   st.write(f"**Size:** {image_data['width']}×{image_data['height']} pixels")
                   st.write(f"**Format:** {image_data['format']}")
                   st.write(f"**File size:** {image_data['size_mb']:.2f} MB")
               
               st.session_state.current_image = image_data
       else:
           st.session_state.current_image = None
       
       st.subheader("Ask Questions or Generate Diagrams")
       
       for message in st.session_state.messages:
           with st.chat_message(message["role"]):
               if message.get("type") == "generated_diagram":
                   if message.get("image_url"):
                       st.image(message["image_url"], caption="Generated Diagram", width=500)
                   
                   st.markdown(message["content"])
                   
               elif message.get("has_image"):
                   col_img, col_text = st.columns([1, 3])
                   with col_img:
                       st.image(message["image"], caption="Context Image", width=150)
                   with col_text:
                       st.markdown(message["content"])
               else:
                   st.markdown(message["content"])
                   
                   if message.get("sources"):
                       with st.expander("View Sources"):
                           for i, source in enumerate(message["sources"], 1):
                               st.text_area(f"Source {i}:", source, height=100, key=f"source_{i}_{message.get('timestamp', 0)}")
       
       prompt = st.chat_input("Ask questions or request diagrams (e.g., 'What is laser interferometry?' or 'Generate a Michelson interferometer diagram')...")
       
       if prompt:
           current_image = st.session_state.current_image
           has_image = current_image is not None
           
           user_message = {
               "role": "user",
               "content": prompt,
               "timestamp": int(time.time())
           }
           
           if has_image:
               user_message.update({
                   "has_image": True,
                   "image": current_image["image"]
               })
           
           st.session_state.messages.append(user_message)
           
           with st.chat_message("user"):
               if has_image:
                   col_img, col_text = st.columns([1, 3])
                   with col_img:
                       st.image(current_image["image"], caption="Context Image", width=150)
                   with col_text:
                       st.markdown(prompt)
               else:
                   st.markdown(prompt)
           
           with st.chat_message("assistant"):
               with st.spinner("Processing..."):
                   image_context = None
                   if has_image:
                       with st.spinner("Analyzing uploaded image..."):
                           image_context = st.session_state.paper_qa.analyze_image(current_image["base64"])
                   
                   if detect_image_generation_request(prompt):
                       with st.spinner("Generating diagram using all available papers..."):
                           
                           try:
                               diagram_url = st.session_state.paper_qa.generate_diagram(prompt)
                               
                               if diagram_url and diagram_url.startswith("http"):
                                   st.image(diagram_url, caption="Generated Diagram", width=500)
                                   response_text = f"""**Generated Displacement Sensing Diagram**

**Prompt:** {prompt}
**Context:** All {len(available_papers)} research papers
{f"**Image Context:** Included analysis from uploaded image" if has_image else ""}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

*Diagram generated with comprehensive research context for maximum accuracy.*"""
                                   
                                   st.session_state.messages.append({
                                       "role": "assistant",
                                       "content": response_text,
                                       "type": "generated_diagram",
                                       "image_url": diagram_url,
                                       "timestamp": int(time.time())
                                   })
                                   
                                   st.markdown(response_text)
                               else:
                                   error_msg = f"**Failed to generate diagram**\n\n{diagram_url}"
                                   st.error(error_msg)
                                   st.session_state.messages.append({
                                       "role": "assistant",
                                       "content": error_msg,
                                       "timestamp": int(time.time())
                                   })
                                   
                           except Exception as e:
                               error_msg = f"Error generating diagram: {str(e)}"
                               st.error(error_msg)
                               st.session_state.messages.append({
                                   "role": "assistant",
                                   "content": error_msg,
                                   "timestamp": int(time.time())
                               })
                           
                   else:
                       with st.spinner("Searching through all research papers..."):
                           try:
                               answer, sources = query_all_papers(st.session_state.paper_qa, prompt)
                               
                               if image_context:
                                   answer = f"**Image Context:** {image_context}\n\n**Answer from Research Papers:**\n{answer}"
                               
                               st.markdown(answer)
                               
                               st.session_state.messages.append({
                                   "role": "assistant",
                                   "content": answer,
                                   "sources": sources,
                                   "timestamp": int(time.time())
                               })
                               
                               if sources:
                                   with st.expander("View Sources"):
                                       for i, source in enumerate(sources, 1):
                                           st.text_area(f"Source {i}:", source, height=100, key=f"live_source_{i}")
                                           
                           except Exception as e:
                               error_msg = f"Error processing question: {str(e)}"
                               st.error(error_msg)
                               st.session_state.messages.append({
                                   "role": "assistant",
                                   "content": error_msg,
                                   "timestamp": int(time.time())
                               })
   
   with col2:
       st.subheader("System Status")
       
       if available_papers:
           st.success(f"{len(available_papers)} papers loaded")
           with st.expander("View Papers"):
               for paper in available_papers:
                   st.write(f"• {paper}")
       else:
           st.error("No papers loaded")
       
       if st.session_state.current_image:
           st.success("Image ready for context")
       else:
           st.info("No image uploaded")
       
       st.divider()
       
       st.subheader("Usage Tips")
       st.markdown("""
       **For Questions:**
       - "What is laser interferometry?"
       - "How do capacitive sensors work?"
       - "Compare optical vs mechanical sensors"
       
       **For Diagrams:**
       - "Generate a Michelson interferometer"
       - "Create LVDT sensor diagram"
       - "Draw capacitive displacement setup"
       
       **With Images:**
       - Upload an image first
       - Then ask questions or request diagrams
       - System will use image for enhanced context
       """)
       
       st.divider()
       
       if st.button("🗑️ Clear Chat", use_container_width=True):
           st.session_state.messages = []
           st.session_state.current_image = None
           st.rerun()

else:
   st.error("OpenAI API key not found in environment variables.")
   st.info("Please make sure your .env file contains OPENAI_API_KEY=your_api_key_here")

st.markdown("---")
st.markdown("Made by humario with love")
