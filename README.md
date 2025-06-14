# Displacement Sensing Research Chatbot 📏

A Streamlit-based chatbot that allows users to query research papers about displacement sensing and generate technical diagrams using.

## How to Use

1. **Ask Questions**: Type questions about displacement sensing (e.g., "What is laser interferometry?")
2. **Generate Diagrams**: Request technical diagrams (e.g., "Generate a Michelson interferometer diagram")
3. **Upload Images**: Add images for enhanced context and analysis
4. **View Sources**: Click "View Sources" to see which papers provided the information

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: LangChain + OpenAI o3-pro/gpt-4o-latest (These are the best models for analysing the images according to the documentation)
- **Vector Database**: ChromaDB
- **Diagram Generation**: gpt-image-1 (this is the most power model for generating images. you can also view the documentation at https://platform.openai.com/docs/models)

## Academic Context

This project is part of a Final Year Project (FYP) focused on displacement sensing research. The chatbot processes multiple research papers to provide comprehensive, research-backed answers about:

- Laser interferometry systems
- Capacitive displacement sensors
- LVDT (Linear Variable Differential Transformer) sensors
- Optical measurement techniques
- Mechanical measurement systems

---

*Made with ❤️ for displacement sensing research*
