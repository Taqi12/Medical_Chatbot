import pandas as pd
import csv

def read_csv_with_error_handling(file_path):
    """
    Reads a CSV file with error handling, preserving all records.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:  # Specify encoding if necessary
        reader = csv.reader(file, quoting=csv.QUOTE_ALL)  # Adjust quoting as needed
        for i, row in enumerate(reader):
            try:
                data.append(row)
            except csv.Error as e:
                print(f"Error reading line {i + 1}: {e}")
                # Handle the error here, e.g., replace problematic values
                # or store them in a separate list for later analysis
                # For now, append the raw row with potential errors
                data.append(row)
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data[1:], columns=data[0]) # Use the first row as column names and data from the second row onwards
    # df.columns = ['column1', 'column2', ...] # This line is no longer needed
    return df

# Call the function to load your CSV
df = read_csv_with_error_handling("/content/medquad 2.csv")

# Display sample data
#print(df.head())

from sentence_transformers import SentenceTransformer, util
import torch
import pickle

# Load SBERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode all questions in dataset
question_embeddings = model.encode(df['question'], convert_to_tensor=True)

# Function to get answer
def get_answer_bert(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, question_embeddings)
    best_match = scores.argmax().item()
    return df.iloc[best_match]['answer']

# Save the SBERT model
model.save("/content/sbert_model")

# Save the encoded question embeddings
torch.save(question_embeddings, "/content/question_embeddings.pt")

# Save the dataset
df.to_pickle("/content/medquad.pkl")

print("✅ Model, embeddings, and dataset saved successfully!")

# Example chatbot interaction
#user_input = "What are the symptoms of diabetes?"
#print(get_answer_bert(user_input))



