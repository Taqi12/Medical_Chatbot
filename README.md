# Medical_Chatbot  

This project is an **AI-powered medical chatbot** that uses **Sentence-BERT (SBERT)** to retrieve the most relevant answers to health-related queries. It matches user questions with a precomputed dataset of medical Q&As.  

## Usage  
- Enter a health-related question in the input field.  
- Click the **"Get Answer"** button.  
- The chatbot will retrieve and display the most relevant answer.  

## Dataset  
[MedQuAD Dataset]([https://www.nlm.nih.gov/databases/download/medquad.html](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research))  
- The model is trained on a **medical question-answer dataset**.  
- The dataset contains **precomputed question embeddings** for efficient retrieval.  

## Model  
- **Sentence-BERT (SBERT)** is used for semantic similarity matching.  
- **Cosine similarity** is used to find the most relevant answer.  
- The chatbot does **not generate** responses but retrieves them from the dataset.  

## Dependencies  
- `streamlit`  
- `torch`  
- `sentence-transformers`  
- `numpy`  
- `pandas`  
- `pickle-mixin`  

## Notes  
- Ensure that `sbert_model`, `question_embeddings.pt`, and `medquad.pkl` are in the root directory.  
- Adjust dataset loading paths in `app.py` if necessary.  
- This chatbot provides **retrieved answers** and is **not a substitute for medical advice**.  

## Author  
[Taqi Javed]  
