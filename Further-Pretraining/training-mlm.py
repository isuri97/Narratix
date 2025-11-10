import torch
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import pandas as pd
import glob
import os
import math
from tqdm import tqdm
import re

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


def extract_sentences_from_text(text):
    """
    Extract individual sentences from text using punctuation marks as delimiters

    Args:
        text: String containing one or more sentences

    Returns:
        List of individual sentences
    """
    if pd.isna(text) or text == '':
        return []

    # Convert to string and clean
    text = str(text).strip()

    # Split on sentence endings (. ! ?) followed by space or end of string
    sentences = re.split(r'[.!?]+[\s]*', text)

    # Clean up sentences - remove empty strings and whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Add punctuation back to sentences (except the last one which might be truncated)
    sentences_with_punct = []
    for i, sentence in enumerate(sentences):
        if sentence and i < len(sentences) - 1:
            # Add period to complete sentences
            sentences_with_punct.append(sentence + '.')
        elif sentence:
            # For the last sentence, check if it already has punctuation
            if not sentence.endswith(('.', '!', '?')):
                sentences_with_punct.append(sentence + '.')
            else:
                sentences_with_punct.append(sentence)

    return sentences_with_punct


def load_and_extract_sentences_from_csv(file_patterns, text_column=None):
    """
    Load texts from multiple CSV files and extract individual sentences

    Args:
        file_patterns: List of file paths or glob patterns
        text_column: Name of the column containing text

    Returns:
        List of individual sentence strings
    """
    all_sentences = []

    for pattern in file_patterns:
        # Handle glob patterns
        files = glob.glob(pattern)

        for file_path in files:
            if os.path.exists(file_path):
                try:
                    print(f"Processing file: {file_path}")

                    # Read CSV file with different encodings if needed
                    try:
                        df = pd.read_csv(file_path)
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='latin-1')

                    # Use specified column or first column if not specified
                    if text_column is None:
                        text_column = df.columns[0]
                    elif text_column not in df.columns:
                        print(f"Column '{text_column}' not found in {file_path}. Available columns: {list(df.columns)}")
                        continue

                    # Extract and process texts
                    texts = df[text_column].dropna().astype(str).tolist()

                    file_sentences = []
                    for text in texts:
                        sentences = extract_sentences_from_text(text)
                        file_sentences.extend(sentences)

                    all_sentences.extend(file_sentences)

                    print(f"Loaded {len(texts)} text entries from {file_path}")
                    print(f"Extracted {len(file_sentences)} individual sentences")
                    if file_sentences:
                        print(f"Sample sentences: {file_sentences[:3]}")

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")

    print(f"Total individual sentences loaded: {len(all_sentences)}")
    return all_sentences


def load_sentences_from_multiple_sources(file_patterns, text_column=None):
    """
    Enhanced function to handle different file types and extract sentences

    Args:
        file_patterns: List of file paths or glob patterns
        text_column: Name of the column containing text (for CSV files)

    Returns:
        List of individual sentence strings
    """
    all_sentences = []

    for pattern in file_patterns:
        files = glob.glob(pattern)

        for file_path in files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            file_extension = os.path.splitext(file_path)[1].lower()

            try:
                if file_extension == '.csv':
                    sentences = process_csv_file(file_path, text_column)
                else:
                    print(f"Unsupported file type: {file_extension} for {file_path}")
                    continue

                all_sentences.extend(sentences)
                print(f"Extracted {len(sentences)} sentences from {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Total sentences extracted from all files: {len(all_sentences)}")
    return all_sentences


def process_csv_file(file_path, text_column=None):
    """Process a single CSV file and extract sentences"""
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')

    if text_column is None:
        text_column = df.columns[0]
    elif text_column not in df.columns:
        print(f"Column '{text_column}' not found. Using first column: {df.columns[0]}")
        text_column = df.columns[0]

    texts = df[text_column].dropna().astype(str).tolist()

    sentences = []
    for text in texts:
        extracted = extract_sentences_from_text(text)
        sentences.extend(extracted)

    return sentences

# def process_text_file(file_path):
#     """Process a single text file and extract sentences"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()
#
#     # Split into paragraphs or lines first
#     paragraphs = content.split('\n\n')
#
#     sentences = []
#     for paragraph in paragraphs:
#         if paragraph.strip():
#             extracted = extract_sentences_from_text(paragraph)
#             sentences.extend(extracted)
#
#     return sentences


# Calculate perplexity
def calculate_perplexity(model, tokenizer, texts, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate perplexity of the model on given texts
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def calculate_perplexity_batch(model, tokenizer, texts, batch_size=8,
                               device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate perplexity in batches (more memory efficient for large datasets)
    """
    model.eval()
    model.to(device)

    total_loss = 0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            for j in range(input_ids.size(0)):
                seq_len = attention_mask[j].sum().item()
                if seq_len > 0:
                    total_loss += outputs.loss.item() * seq_len
                    total_tokens += seq_len

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')

    return perplexity, avg_loss

# Example usage for multiple files
csv_files = [
    "data/*.csv",
]

text_column = "content"  # Change to your column name for CSV files

# Load and extract sentences from multiple files
print("Loading and extracting sentences from files...")
sentences = load_sentences_from_multiple_sources(csv_files, text_column)

# If no sentences loaded, use sample data
if not sentences:
    print("Using sample data as fallback")
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized the field of NLP.",
        "BERT is a powerful pre-trained model for various NLP tasks."
    ]

print(f"\nFinal dataset: {len(sentences)} individual sentences")

# Split data into train and evaluation sets
train_size = int(0.8 * len(sentences))
train_sentences = sentences[:train_size]
eval_sentences = sentences[train_size:]

print(f"Training sentences: {len(train_sentences)}")
print(f"Evaluation sentences: {len(eval_sentences)}")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Create datasets
train_dataset = Dataset.from_dict({"text": train_sentences})
eval_dataset = Dataset.from_dict({"text": eval_sentences})

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments with evaluation
training_args = TrainingArguments(
    output_dir="./bert-mlm-results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=100,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Calculate initial perplexity
print("\nCalculating initial perplexity...")
initial_perplexity, initial_loss = calculate_perplexity_batch(model, tokenizer, eval_sentences)
print(f"Initial Perplexity: {initial_perplexity:.4f}")
print(f"Initial Loss: {initial_loss:.4f}")

# Train the model
print("\nStarting training...")
trainer.train()

# Save the model
trainer.save_model("./bert-mlm-model")
tokenizer.save_pretrained("./bert-mlm-model")

# Calculate final perplexity
print("\nCalculating final perplexity...")
final_perplexity, final_loss = calculate_perplexity_batch(model, tokenizer, eval_sentences)
print(f"Final Perplexity: {final_perplexity:.4f}")
print(f"Final Loss: {final_loss:.4f}")
print(f"Perplexity Improvement: {initial_perplexity - final_perplexity:.4f}")

# Append final results to text file
with open("training_results.txt", "a") as f:
    f.write("\n=== Training Completed ===\n")
    f.write(f"Final Perplexity: {final_perplexity:.4f}\n")
    f.write(f"Final Loss: {final_loss:.4f}\n")
    f.write(f"Perplexity Improvement: {initial_perplexity - final_perplexity:.4f}")
    f.write(f"Training epochs: {training_args.num_train_epochs}\n")
    f.write(f"Model saved to: ./bert-mlm-model\n")
