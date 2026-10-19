import csv
import fitz  # PyMuPDF
import faiss
import torch
import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load CSV data using built-in csv module
def load_symptom_condition_map(filepath):
    mapping = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symptom = row['symptom'].strip().lower()
            condition = row['condition'].strip().lower()
            mapping.setdefault(symptom, []).append(condition)
    return mapping

def load_condition_treatment_map(filepath):
    mapping = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row['condition'].strip().lower()
            treatment = row['treatment'].strip()
            precaution = row['precaution'].strip()
            mapping[condition] = {'treatment': treatment, 'precaution': precaution}
    return mapping

# PDF text processing
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

def chunk_text(text, max_tokens=100):
    sentences = text.split('. ')
    chunks, chunk, token_count = [], [], 0
    for sentence in sentences:
        tokens = sentence.split()
        if token_count + len(tokens) > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0
        chunk.append(sentence)
        token_count += len(tokens)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Embedding setup
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(text_chunks):
    embeddings = embedder.encode(text_chunks, convert_to_tensor=True).cpu().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Load LLM for generation
generator_tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
generator_model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

def get_context(query, chunks, index, embedder, top_k=1):
    query_vec = embedder.encode([query], convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

def generate_response(user_input, chunks, index, embedder):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon']
    farewells = ['bye', 'goodbye', 'see you', 'take care']
    if any(g in user_input.lower() for g in greetings):
        return "Hello! How can I assist you today?"
    if any(f in user_input.lower() for f in farewells):
        return "Goodbye! Take care and feel free to reach out if you need assistance."

    # Check symptom -> condition -> treatment
    matched_conditions = []
    for symptom, conditions in symptom_condition_map.items():
        if symptom in user_input.lower():
            matched_conditions.extend(conditions)

    if matched_conditions:
        matched_conditions = list(set(matched_conditions))[:3]  # Limit to 3
        responses = []
        for cond in matched_conditions:
            treatment_info = condition_treatment_map.get(cond, {})
            if treatment_info:
                responses.append(f"- **{cond.capitalize()}**: Treatment - {treatment_info['treatment']}. Precaution - {treatment_info['precaution']}")
            else:
                responses.append(f"- **{cond.capitalize()}**: No detailed info available.")
        return "Based on your symptoms, here are possible conditions and recommendations:\n" + "\n".join(responses)

    # Use PDF fallback
    context = get_context(user_input, chunks, index, embedder)[0]
    prompt = f"Patient: {user_input}\nDoctor: {context}\nAssistant:"
    input_ids = generator_tokenizer.encode(prompt, return_tensors='pt')
    output_ids = generator_model.generate(input_ids, max_new_tokens=150, pad_token_id=generator_tokenizer.pad_token_id)
    response = generator_tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

# Tkinter GUI
def start_chatbot(chunks, index):
    def send_message():
        user_msg = user_entry.get()
        if not user_msg.strip():
            return
        chat_area.insert(tk.END, "You: " + user_msg + "\n")
        bot_response = generate_response(user_msg, chunks, index, embedder)
        if not bot_response:
            bot_response = "I'm sorry, I couldn't find relevant information."
        chat_area.insert(tk.END, "Bot: " + bot_response + "\n\n")
        user_entry.delete(0, tk.END)

    root = tk.Tk()
    root.title("AI Medical Chatbot")
    root.geometry("600x600")

    chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=25, font=("Arial", 10))
    chat_area.pack(pady=10)

    user_entry = tk.Entry(root, width=50, font=("Arial", 12))
    user_entry.pack(padx=10, pady=5, side=tk.LEFT)

    send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 12))
    send_button.pack(padx=10, pady=5, side=tk.RIGHT)

    root.mainloop()

if __name__ == "__main__":
    print("Loading PDF and building chatbot...")
    symptom_condition_map = load_symptom_condition_map("symptoms_conditions.csv")
    condition_treatment_map = load_condition_treatment_map("conditions_treatments.csv")
    pdf_text = extract_text_from_pdf("Medical_book.pdf")
    text_chunks = chunk_text(pdf_text)
    index, _ = build_faiss_index(text_chunks)
    print("Starting chatbot...")
    start_chatbot(text_chunks, index)
