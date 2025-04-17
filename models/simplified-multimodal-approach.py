# -*- coding: utf-8 -*-
"""
Multi-stage approach: Extract features + fine-tune Gemma with image features as text tokens
"""

# 1. Import standard libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that doesn't require X11
import matplotlib.pyplot as plt
from PIL import Image
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize

# 2. Set NLTK data path early
import nltk
nltk.data.path.append('/home/h.chehili/models2025nourhene/nltk_data')

# 3. Define output directory
base_dir = "/home/h.chehili/models2025nourhene"
output_dir = f"{base_dir}/output/gemma_multimodal"
os.makedirs(output_dir, exist_ok=True)

# Set up evaluation directory 
eval_dir = f"{output_dir}/evaluation"
os.makedirs(eval_dir, exist_ok=True)

# Start logging
logfile = f"{output_dir}/gemma_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message):
    print(message)
    with open(logfile, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

log_message(f"Starting script execution with PyTorch {torch.__version__}")
log_message(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log_message(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log_message(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 4. Load the CSV data with error handling
filtered_csv_path = "/home/h.chehili/models2025nourhene/OpenI_Project/openi_dataset/filtered_resultat.csv"
try:
    log_message(f"Loading CSV from: {filtered_csv_path}")
    filtered_dataset = pd.read_csv(
        filtered_csv_path,
        engine='python',      # Use more flexible Python engine
        on_bad_lines='skip',  # Skip problematic lines
        quotechar='"',        # Specify quote character
        escapechar='\\'       # Specify escape character
    )
    log_message(f"Successfully loaded CSV with {len(filtered_dataset)} rows")
except Exception as e:
    log_message(f"Error with first CSV loading method: {str(e)}")
    log_message("Trying alternative loading method...")
    filtered_dataset = pd.read_csv(
        filtered_csv_path,
        engine='python',
        sep=None,             # Auto-detect separator
        header=0
    )
    log_message(f"Alternative method loaded {len(filtered_dataset)} rows")

# Check if required columns exist
required_columns = ['Valid_Images', 'Findings', 'Impressions', 'Indications', 'subset']
missing_columns = [col for col in required_columns if col not in filtered_dataset.columns]
if missing_columns:
    log_message(f"WARNING: Missing required columns: {missing_columns}")
    log_message(f"Available columns: {filtered_dataset.columns.tolist()}")

# Fill missing values to avoid errors
filtered_dataset = filtered_dataset.fillna({
    'Findings': '',
    'Impressions': '',
    'Indications': ''
})

# 5. Set up device (using a single specific GPU to avoid device mismatches)
target_gpu = 0  # Use GPU 0
device = torch.device(f'cuda:{target_gpu}' if torch.cuda.is_available() else 'cpu')
log_message(f"Using device: {device}")

# 6. Define image preprocessing and dataset for feature extraction
img_size = (224, 224)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MedicalImageDataset(Dataset):
    def __init__(self, dataframe, img_col="Valid_Images", transform=None):
        self.dataframe = dataframe
        self.img_col = img_col
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        try:
            img_path = self.dataframe.iloc[idx][self.img_col]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image
        except Exception as e:
            log_message(f"Error loading image at index {idx}: {str(e)}")
            # Return a blank image instead of failing
            blank = torch.zeros(3, *img_size)
            return blank

# 7. Load pre-trained PyTorch models for feature extraction
log_message("Loading feature extraction models...")

try:
    # ResNet50 - updated to use weights parameter instead of pretrained
    resnet_model = models.resnet50(weights=None)
    resnet_model.load_state_dict(torch.load(f"{base_dir}/pytorch/resnet50.pth", weights_only=True))
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    log_message("ResNet50 loaded successfully")

    # EfficientNetB0 - updated to use weights parameter instead of pretrained
    efficientnet_model = models.efficientnet_b0(weights=None)
    efficientnet_model.load_state_dict(torch.load(f"{base_dir}/pytorch/efficientnet_b0.pth", weights_only=True))
    efficientnet_model = efficientnet_model.to(device)
    efficientnet_model.eval()
    log_message("EfficientNetB0 loaded successfully")
except Exception as e:
    log_message(f"Error loading feature extraction models: {str(e)}")
    raise

# Check if preprocessed features already exist
features_path = f"{output_dir}/final_features.npy"
if os.path.exists(features_path):
    log_message(f"Loading pre-extracted features from {features_path}")
    final_features = np.load(features_path)
    log_message(f"Loaded features with shape: {final_features.shape}")
else:
    # Create dataset and dataloader for feature extraction
    try:
        image_dataset = MedicalImageDataset(filtered_dataset, transform=transform)
        dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
        log_message(f"Created dataloader with {len(image_dataset)} images")
    except Exception as e:
        log_message(f"Error creating dataloader: {str(e)}")
        raise

    # 8. Extract image features
    def extract_features(model, dataloader):
        features = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i % 10 == 0:
                    log_message(f"Processing batch {i}/{len(dataloader)}")
                    
                batch = batch.to(device)
                
                # For ResNet, we need to remove the classification layer
                if isinstance(model, models.ResNet):
                    batch_features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(batch)))))))))
                    batch_features = batch_features.reshape(batch_features.size(0), -1)
                # For EfficientNet
                else:
                    batch_features = model.features(batch)
                    batch_features = model.avgpool(batch_features)
                    batch_features = batch_features.reshape(batch_features.size(0), -1)
                
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features)

    # Extract features
    log_message("Extracting image features...")
    features_resnet_all = extract_features(resnet_model, dataloader)
    features_efficient_all = extract_features(efficientnet_model, dataloader)

    log_message(f"Shape of ResNet features: {features_resnet_all.shape}")
    log_message(f"Shape of EfficientNet features: {features_efficient_all.shape}")

    # 9. Process text using BERT
    log_message("Processing text with BERT...")
    from transformers import BertTokenizer, BertModel

    # Load BERT tokenizer and model
    bert_tokenizer = BertTokenizer.from_pretrained(
        f"{base_dir}/transformers/bert-tokenizer",
        local_files_only=True
    )
    bert_model = BertModel.from_pretrained(
        f"{base_dir}/transformers/bert-model",
        local_files_only=True
    ).to(device)
    bert_model.eval()

    def preprocess_text_pytorch(texts, max_length=128, batch_size=32):
        """Process text with PyTorch BERT"""
        all_embeddings = []
        # Convert all elements to strings
        texts = [str(text) for text in texts]
        
        for i in range(0, len(texts), batch_size):
            if i % (batch_size * 10) == 0:
                log_message(f"Processing text batch {i//batch_size}/{len(texts)//batch_size}")
                
            batch_texts = texts[i:i+batch_size]
            inputs = bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                # Extract CLS token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)

    # Process findings and impressions
    findings_all = filtered_dataset["Findings"].tolist()
    impressions_all = filtered_dataset["Impressions"].tolist()

    findings_embeddings_all = preprocess_text_pytorch(findings_all)
    impressions_embeddings_all = preprocess_text_pytorch(impressions_all)

    # Concatenate text embeddings
    text_embeddings_all = np.concatenate([findings_embeddings_all, impressions_embeddings_all], axis=1)

    # Normalize embeddings
    text_embeddings_all = normalize(text_embeddings_all, axis=1)

    log_message(f"Text embeddings shape: {text_embeddings_all.shape}")

    # 10. Combine visual and text features
    final_features = np.concatenate((features_resnet_all, features_efficient_all, text_embeddings_all), axis=1)
    log_message(f"Final features shape: {final_features.shape}")

    # 11. Save extracted features for future use
    np.save(f"{output_dir}/final_features.npy", final_features)
    np.save(f"{output_dir}/features_resnet.npy", features_resnet_all)
    np.save(f"{output_dir}/features_efficient.npy", features_efficient_all)
    np.save(f"{output_dir}/text_embeddings.npy", text_embeddings_all)
    log_message("Saved extracted features to disk")

# 12. STAGE 2: Fine-tune Gemma (with text-encoded image features)
log_message("Starting Stage 2: Gemma fine-tuning with encoded image features...")

# Set environment variables for distributed training
import os
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Load Gemma tokenizer
gemma_model_path = f"{base_dir}/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(
    gemma_model_path,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

# Function to encode image features as text
def encode_image_features(features, num_features=20):
    """Convert image features to text representation"""
    # Select a subset of features for readability
    subset_features = features[:num_features]
    # Round to 3 decimal places and convert to string
    return ", ".join([f"{val:.3f}" for val in subset_features])

# Function to format instruction tuning format for Gemma
def format_gemma_prompt(instruction, input_text, output=""):
    prompt = f"<start_of_turn>user\n{instruction}"
    if input_text:
        prompt += f"\n\n{input_text}<end_of_turn>\n"
    else:
        prompt += "<end_of_turn>\n"
        
    if output:
        prompt += f"<start_of_turn>model\n{output}<end_of_turn>"
    else:
        prompt += "<start_of_turn>model\n"
        
    return prompt

# Create a dataset with encoded image features
class GemmaMultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_features, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.image_features = image_features
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Get corresponding image features
        img_features = self.image_features[idx]
        
        # Encode image features as text (use first 10 from each model)
        resnet_size = 2048  # Typical ResNet feature size
        efficient_size = 1280  # Typical EfficientNet feature size
        
        resnet_features = img_features[:resnet_size]
        efficient_features = img_features[resnet_size:resnet_size+efficient_size]
        
        # Encode a sample of features as text (first 10 from each model)
        resnet_text = encode_image_features(resnet_features, 10)
        efficient_text = encode_image_features(efficient_features, 10)
        
        # Create instruction format with image features
        instruction = "Generate an impression for a medical radiograph based on the findings, indications, and image features."
        input_text = (
            f"Findings: {row['Findings']}\n"
            f"Indications: {row['Indications']}\n"
            f"ResNet features: {resnet_text}\n"
            f"EfficientNet features: {efficient_text}"
        )
        output = row['Impressions']
        
        # Format for Gemma
        prompt = format_gemma_prompt(instruction, input_text, output)
        
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            prompt, 
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Get tokens
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        # Set labels for loss calculation (only for the model output part)
        model_start_token = self.tokenizer.encode("<start_of_turn>model", add_special_tokens=False)[0]
        model_start_pos = (input_ids == model_start_token).nonzero(as_tuple=True)[0]
        if len(model_start_pos) > 0:
            model_start_pos = model_start_pos[-1]  # In case there are multiple occurrences
        else:
            model_start_pos = 0
        
        # Create labels: -100 for tokens we don't want to calculate loss on
        labels = input_ids.clone()
        labels[:model_start_pos] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Split data into training and validation sets
train_df = filtered_dataset[filtered_dataset['subset'] == 'training']
val_df = filtered_dataset[filtered_dataset['subset'] == 'validation']

# Filter data to include only valid samples (with valid findings and impressions)
train_df = train_df[
    (train_df['Findings'].notna()) & 
    (train_df['Impressions'].notna()) & 
    (train_df['Impressions'].str.len() > 10)
]
val_df = val_df[
    (val_df['Findings'].notna()) & 
    (val_df['Impressions'].notna()) & 
    (val_df['Impressions'].str.len() > 10)
]

log_message(f"Prepared {len(train_df)} training samples and {len(val_df)} validation samples")

# Get indices for train and validation sets
train_indices = []
for idx in train_df.index:
    train_indices.append(filtered_dataset.index.get_loc(idx))

val_indices = []
for idx in val_df.index:
    val_indices.append(filtered_dataset.index.get_loc(idx))

# Extract features for train and validation sets
train_features = final_features[train_indices]
val_features = final_features[val_indices]

# Create datasets with image features
train_dataset = GemmaMultimodalDataset(train_df, tokenizer, train_features)
val_dataset = GemmaMultimodalDataset(val_df, tokenizer, val_features)

# Load Gemma model with LoRA
log_message("Loading Gemma model for fine-tuning...")
model = AutoModelForCausalLM.from_pretrained(
    gemma_model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map={"": target_gpu}  # Force on single GPU
)

# Configure LoRA for parameter-efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
log_message("Model prepared with LoRA adapters")

# Set up training arguments
training_args = TrainingArguments(
    output_dir=f"{output_dir}/checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_steps=100,
    save_steps=200,
    warmup_steps=100,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    eval_strategy="steps",  # Use the parameter name that works with your version
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="none",  # Disable online tracking
    # Explicitly disable distributed training parameters
    local_rank=-1,
    ddp_backend=None,
    dataloader_num_workers=0,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Check if we already have a trained model
model_path = f"{output_dir}/gemma-lora-final"
if os.path.exists(model_path):
    log_message(f"Found existing trained model at {model_path}. Skipping training.")
    # Load the pre-trained model
    from peft import PeftModel
    
    base_model = AutoModelForCausalLM.from_pretrained(
        gemma_model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": target_gpu}
    )
    model = PeftModel.from_pretrained(base_model, model_path)
else:
    # Train model
    log_message("Starting Gemma fine-tuning with encoded image features...")
    try:
        trainer.train()
        
        # Save the final model
        log_message("Training completed, saving model...")
        model.save_pretrained(f"{output_dir}/gemma-lora-final")
        tokenizer.save_pretrained(f"{output_dir}/gemma-lora-final")
        log_message("Model saved successfully")
    except Exception as e:
        log_message(f"Error during training: {str(e)}")
        model.save_pretrained(f"{output_dir}/gemma-lora-interrupted")
        tokenizer.save_pretrained(f"{output_dir}/gemma-lora-interrupted")
        log_message(f"Partially trained model saved to {output_dir}/gemma-lora-interrupted")
        raise

# 13. Generate function for testing with encoded image features
def generate_multimodal_report(findings, indications, image_feature, max_length=150):
    # Encode image features as text
    resnet_size = 2048  # Typical ResNet feature size
    efficient_size = 1280  # Typical EfficientNet feature size
    
    resnet_features = image_feature[:resnet_size]
    efficient_features = image_feature[resnet_size:resnet_size+efficient_size]
    
    # Encode a sample of features as text
    resnet_text = encode_image_features(resnet_features, 10)
    efficient_text = encode_image_features(efficient_features, 10)
    
    # Create instruction
    instruction = "Generate an impression for a medical radiograph based on the findings, indications, and image features."
    input_text = (
        f"Findings: {findings}\n"
        f"Indications: {indications}\n"
        f"ResNet features: {resnet_text}\n"
        f"EfficientNet features: {efficient_text}"
    )
    
    # Format for Gemma
    prompt = format_gemma_prompt(instruction, input_text)
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
    
    # Decode and extract the model's response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the model's response
    response = full_text.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
    return response

# 14. Test the model
try:
    log_message("Testing multimodal medical report generation...")
    # Sample from the validation set
    sample_idx = 0
    sample = val_df.iloc[sample_idx]
    findings = sample["Findings"]
    indications = sample["Indications"]
    
    # Get corresponding image features
    val_idx = val_indices[sample_idx]
    image_feature = final_features[val_idx]
    
    generated_impression = generate_multimodal_report(findings, indications, image_feature)
    
    log_message(f"\nSample Findings: {findings}")
    log_message(f"\nSample Indications: {indications}")
    log_message(f"\nGenerated Impression: {generated_impression}")
    log_message(f"\nReference Impression: {sample['Impressions']}")
    
except Exception as e:
    log_message(f"Error testing model: {str(e)}")

# 15. Comprehensive model evaluation with image features
log_message("Starting comprehensive model evaluation with image features...")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge

# Evaluate on validation set (limited sample for speed)
eval_samples = min(50, len(val_df))
log_message(f"Evaluating on {eval_samples} samples from validation set")

results = []
for i in range(eval_samples):
    if i % 5 == 0:
        log_message(f"Processing evaluation sample {i+1}/{eval_samples}")
        
    try:
        # Get sample data
        sample = val_df.iloc[i]
        findings = sample["Findings"]
        indications = sample["Indications"]
        reference = sample["Impressions"]
        
        # Get image features
        val_idx = val_indices[i]
        image_feature = final_features[val_idx]
        
        # Generate impression with image features
        generated = generate_multimodal_report(findings, indications, image_feature)
        
        # Save result
        results.append({
            'findings': findings,
            'indications': indications,
            'reference': reference,
            'generated': generated
        })
        
        # Log a successful generation occasionally
        if i % 10 == 0:
            log_message(f"Successfully generated report {i+1}:")
            log_message(f"Generated: {generated[:100]}...")
            log_message(f"Reference: {reference[:100]}...")
            
    except Exception as e:
        log_message(f"Error processing evaluation sample {i}: {str(e)}")

# Save all results to a file
with open(f"{eval_dir}/generation_results.txt", "w") as f:
    for i, result in enumerate(results):
        f.write(f"Sample {i+1}:\n")
        f.write(f"Findings: {result['findings']}\n")
        f.write(f"Indications: {result['indications']}\n")
        f.write(f"Reference: {result['reference']}\n")
        f.write(f"Generated: {result['generated']}\n")
        f.write("-" * 80 + "\n\n")

# Calculate metrics
log_message("Calculating evaluation metrics...")
try:
    # Initialize metric calculations
    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    length_ratios = []
    rouge = Rouge()
    
    for result in results:
        reference = result['reference']
        generated = result['generated']
        
        # Skip empty results
        if not reference or not generated:
            continue
            
        # BLEU score
        try:
            smoothie = SmoothingFunction().method4
            reference_tokens = [reference.split()]
            generated_tokens = generated.split()
            bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)
        except Exception as e:
            log_message(f"Error calculating BLEU: {str(e)}")
        
        # ROUGE score
        try:
            rouge_score = rouge.get_scores(generated, reference)[0]
            rouge_scores.append(rouge_score)
        except Exception as e:
            log_message(f"Error calculating ROUGE: {str(e)}")
        
        # METEOR score - try but don't fail if it fails
        try:
            tokenized_generated = word_tokenize(generated)
            tokenized_reference = [word_tokenize(reference)]
            meteor_score_value = meteor_score(tokenized_reference, tokenized_generated)
            meteor_scores.append(meteor_score_value)
        except Exception as e:
            log_message(f"Error calculating METEOR (skipping): {str(e)}")
        
        # Length ratio (generated / reference)
        length_ratio = len(generated.split()) / max(1, len(reference.split()))
        length_ratios.append(length_ratio)
    
    # Calculate average metrics
    metrics = {}
    if bleu_scores:
        metrics['bleu'] = np.mean(bleu_scores)
    if rouge_scores:
        metrics['rouge-1'] = np.mean([s['rouge-1']['f'] for s in rouge_scores])
        metrics['rouge-2'] = np.mean([s['rouge-2']['f'] for s in rouge_scores])
        metrics['rouge-l'] = np.mean([s['rouge-l']['f'] for s in rouge_scores])
    if meteor_scores:
        metrics['meteor'] = np.mean(meteor_scores)
    if length_ratios:
        metrics['length_ratio'] = np.mean(length_ratios)
    
    # Log and save metrics
    log_message("Evaluation metrics:")
    for metric, value in metrics.items():
        log_message(f"{metric}: {value:.4f}")
    
    with open(f"{eval_dir}/metrics.txt", "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Create visualization of the metrics
    plt.figure(figsize=(10, 6))
    metric_names = []
    metric_values = []
    
    for metric in ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'meteor']:
        if metric in metrics:
            metric_names.append(metric)
            metric_values.append(metrics[metric])
    
    plt.bar(metric_names, metric_values)
    plt.title('Evaluation Metrics - Gemma with Encoded Image Features')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig(f"{eval_dir}/metrics_visualization.png")
    log_message(f"Metrics visualization saved to {eval_dir}/metrics_visualization.png")
    
except Exception as e:
    log_message(f"Error in metrics calculation: {str(e)}")

# Compare with original model (if available)
try:
    original_metrics_path = f"{base_dir}/output/gemma/evaluation/metrics.txt"
    if os.path.exists(original_metrics_path):
        log_message("Comparing with original model results...")
        original_metrics = {}
        with open(original_metrics_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    original_metrics[key.strip()] = float(value.strip())
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        # Find common metrics
        common_metrics = []
        multimodal_values = []
        original_values = []
        
        for metric in ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'meteor']:
            if metric in metrics and metric in original_metrics:
                common_metrics.append(metric)
                multimodal_values.append(metrics[metric])
                original_values.append(original_metrics[metric])
        
        # Set up bar positions
        x = np.arange(len(common_metrics))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, multimodal_values, width, label='With Image Features', color='blue')
        plt.bar(x + width/2, original_values, width, label='Text-Only Model', color='orange')
        
        # Add labels and title
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Metric Comparison: With vs. Without Image Features')
        plt.xticks(x, common_metrics)
        plt.legend()
        
        # Add values on top of bars
        for i, v in enumerate(multimodal_values):
            plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
        
        for i, v in enumerate(original_values):
            plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.savefig(f"{eval_dir}/comparison_visualization.png")
        log_message(f"Comparison visualization saved to {eval_dir}/comparison_visualization.png")
except Exception as e:
    log_message(f"Error comparing with original model: {str(e)}")

log_message("Multi-stage processing with encoded image features completed!")
