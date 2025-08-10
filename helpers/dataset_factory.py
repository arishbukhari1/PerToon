"""
Unified Dataset Factory for PerToon Project
=============================================================================

This module provides a unified interface for loading and processing datasets.
It now fully encapsulates the ImgFlip pipeline, from download to sampling to labeling,
with configurable file paths for robust caching at each step.

Key Features:
- Handles pre-labeled (GoEmotions) and unlabeled (ImgFlip) datasets
- Unified pipeline: Load → Label → EDA → Balance → Format → Tokenize
- Configuration-driven dataset switching with configurable paths
- Robust caching at each stage (raw, sampled, labeled)
- Reusable across 2a, 2c, and future notebooks
"""

import pandas as pd
import torch
import os
from datasets import load_dataset, Dataset
from transformers import pipeline, GPT2Tokenizer, DataCollatorForLanguageModeling
import requests
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Unified emotion labels (aligned with GoEmotions)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

class DatasetConfig:
    """Configuration class for different datasets."""
    
    def __init__(self, dataset_name: str, has_labels: bool, loader_function: str, **kwargs):
        self.dataset_name = dataset_name
        self.has_labels = has_labels
        self.loader_function = loader_function
        self.kwargs = kwargs
        # Set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- NEW: Dataset configurations with configurable paths ---
DATASET_CONFIGS = {
    "goemotions": DatasetConfig(
        dataset_name="goemotions",
        has_labels=True,
        loader_function="load_goemotions",
        labeled_path="mood_captions_goemotions.csv"
    ),
    "imgflip": DatasetConfig(
        dataset_name="imgflip", 
        has_labels=False,
        loader_function="load_imgflip",
        labeler_model="facebook/bart-large-mnli",
        # Default paths for the full dataset processing
        raw_path="captions_imgflip.csv",
        sampled_path="captions_imgflip_full_sample.csv",
        labeled_path="mood_captions_imgflip.csv",
        # Default processing parameters
        captions_to_process=200000,
        batch_size=100
    )
}

class UnifiedDatasetFactory:
    """
    Factory class that provides a unified interface for loading and processing
    different datasets with emotion labels.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the dataset factory.
        
        Args:
            project_root (str, optional): Root path of the project
        """
        self.project_root = project_root or ".."
        self.data_dir = f"{self.project_root}/data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_dataset(self, dataset_name: str, **override_kwargs) -> pd.DataFrame:
        """
        Load and process a dataset with unified output format.
        
        Args:
            dataset_name (str): Name of dataset ("goemotions" or "imgflip")
            **override_kwargs: Override default configuration parameters (including file paths)
            
        Returns:
            pd.DataFrame: Processed dataframe with 'mood' and 'caption' columns
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
            
        config = DATASET_CONFIGS[dataset_name]
        
        # Override config with any provided kwargs (including file paths)
        for key, value in override_kwargs.items():
            if value is not None:
                setattr(config, key, value)
            
        print(f"Loading {config.dataset_name} dataset...")
        
        # Call the appropriate loader function
        loader_method = getattr(self, config.loader_function)
        df = loader_method(config)
        
        # Ensure output format consistency
        if 'mood' not in df.columns or 'caption' not in df.columns:
            raise ValueError("Loader must return DataFrame with 'mood' and 'caption' columns.")
            
        return df
    
    def load_goemotions(self, config: DatasetConfig) -> pd.DataFrame:
        """
        Load GoEmotions dataset (pre-labeled).
        
        Args:
            config (DatasetConfig): Dataset configuration
            
        Returns:
            pd.DataFrame: Processed dataframe with mood and caption columns
        """
        print("    Loading GoEmotions from HuggingFace...")
        
        # Load GoEmotions dataset
        goemotions = load_dataset("go_emotions", "simplified", split="train")
        df = goemotions.to_pandas()
        
        print(f"    Original dataset size: {len(df)} samples")
        
        # Keep samples with only one label (cleaner training data)
        df["num_labels"] = df["labels"].apply(len)
        df_single_label = df[df["num_labels"] == 1].copy()
        
        print(f"    Single-label samples: {len(df_single_label)} samples")
        
        # Map label integers to emotion names
        label_names = goemotions.features["labels"].feature.names
        df_single_label["mood"] = df_single_label["labels"].apply(lambda x: label_names[x[0]])
        df_single_label["caption"] = df_single_label["text"]
        
        # Select relevant features
        final_df = df_single_label[["mood", "caption"]].copy()
        
        print(f"    GoEmotions processed: {len(final_df)} samples with {final_df['mood'].nunique()} emotions")
        return final_df
    
    # --- HEAVILY REFACTORED: This function now contains the full pipeline logic ---
    def load_imgflip(self, config: DatasetConfig) -> pd.DataFrame:
        """
        Orchestrates the loading of the ImgFlip dataset, using cached files at each stage.
        """
        # Define full paths from config
        labeled_path = os.path.join(self.data_dir, config.labeled_path)
        sampled_path = os.path.join(self.data_dir, config.sampled_path)
        raw_path = os.path.join(self.data_dir, config.raw_path)

        # 1. Check for the final, labeled file
        if os.path.exists(labeled_path):
            print(f"    Found pre-labeled data. Loading from: {labeled_path}")
            df = pd.read_csv(labeled_path)
            if 'predicted_emotion' in df.columns:
                df = df.rename(columns={'predicted_emotion': 'mood'})
            return df[['mood', 'caption']]

        # 2. If no labeled file, check for a sampled (but unlabeled) file
        if os.path.exists(sampled_path):
            print(f"    Found pre-sampled data. Loading from: {sampled_path}")
            sampled_df = pd.read_csv(sampled_path)
        else:
            # 3. If no sampled file, get raw captions (download if needed)
            if os.path.exists(raw_path):
                print(f"    Found raw captions. Loading from: {raw_path}")
                raw_df = pd.read_csv(raw_path)
                if 'caption' not in raw_df.columns and raw_df.shape[1] == 1:
                    raw_df.columns = ['caption']
            else:
                print("    No raw captions found. Downloading from GitHub...")
                raw_df = self._download_imgflip_captions(raw_path)
            
            # 4. Sample the raw captions
            print(f"    Sampling {config.captions_to_process} captions...")
            sampled_df = self._sample_captions(raw_df, config.captions_to_process, sampled_path)
        
        # 5. Label the sampled captions
        print("    Applying zero-shot emotion labeling...")
        labeled_df = self._apply_emotion_labeling(
            sampled_df,
            labeled_path,
            config.labeler_model,
            config.batch_size
        )

        print(f"    ImgFlip processing complete: {len(labeled_df)} samples with {labeled_df['mood'].nunique()} emotions.")
        return labeled_df
    
    def _download_imgflip_captions(self, save_path: str) -> pd.DataFrame:
        """Download ImgFlip captions from GitHub repository."""
        
        # If file already exists, don't re-download
        if os.path.exists(save_path):
            print(f"       Loading existing captions from {save_path}")
            try:
                df = pd.read_csv(save_path)
                if 'caption' not in df.columns and df.shape[1] == 1:
                    df.columns = ['caption']
                print(f"       Loaded {len(df)} captions from existing file")
                return df
            except Exception as e:
                print(f"       Error reading existing file: {e}, downloading from GitHub...")
        
        # Download from GitHub
        api_url = "https://api.github.com/repos/schesa/ImgFlip575K_Dataset/contents/dataset/memes"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            files = response.json()
            
            all_captions = []
            print(f"       Downloading from GitHub repo...")
            
            for file_info in files:
                if file_info['name'].endswith('.json'):
                    print(f"         Processing {file_info['name']}...")
                    
                    file_response = requests.get(file_info['download_url'])
                    file_response.raise_for_status()
                    data = file_response.json()
                    
                    # Handle both single objects and lists
                    if not isinstance(data, list):
                        data = [data]
                    
                    for meme in data:
                        boxes = meme.get('boxes', [])
                        if boxes:
                            all_captions.append(' '.join(boxes))
            
            captions_df = pd.DataFrame({'caption': all_captions})
            captions_df.to_csv(save_path, index=False)
            print(f"       Saved {len(captions_df)} raw captions to {save_path}")
            return captions_df
            
        except Exception as e:
            print(f"       Error downloading ImgFlip data: {e}")
            return pd.DataFrame({'caption': []})
    
    # --- NEW: This method encapsulates the sampling logic from the notebook ---
    def _sample_captions(self, df: pd.DataFrame, num_to_sample: int, save_path: str) -> pd.DataFrame:
        """Samples captions from a DataFrame and saves them to a CSV."""
        shuffled_df = df.dropna(subset=['caption']).sample(frac=1, random_state=42)
        sampled_df = shuffled_df.head(num_to_sample)[['caption']].reset_index(drop=True)
        sampled_df.to_csv(save_path, index=False)
        print(f"       Saved {len(sampled_df)} sampled captions to {save_path}")
        return sampled_df
    
    # --- MODIFIED: Accepts the DataFrame to label and the output path ---
    def _apply_emotion_labeling(self, df_to_label: pd.DataFrame, save_path: str, model_name: str, batch_size: int) -> pd.DataFrame:
        """Applies zero-shot emotion labeling and saves the result."""
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        
        print(f"       Processing {len(df_to_label)} captions in batches of {batch_size}...")
        
        labeled_rows = []
        from tqdm import tqdm
        for i in tqdm(range(0, len(df_to_label), batch_size), desc="Labeling batches"):
            batch = df_to_label.iloc[i:i+batch_size]['caption'].tolist()
            results = classifier(batch, EMOTION_LABELS, multi_label=False)
            
            if not isinstance(results, list):
                results = [results]
                
            for j, res in enumerate(results):
                labeled_rows.append({
                    "caption": batch[j], 
                    "mood": res['labels'][0], 
                    "confidence": res['scores'][0]
                })
        
        labeled_df = pd.DataFrame(labeled_rows)
        # Filter by confidence and select columns
        labeled_df = labeled_df[labeled_df['confidence'] >= 0.3][['mood','caption']]
        
        # Save the final labeled data
        labeled_df.to_csv(save_path, index=False)
        print(f"       Saved {len(labeled_df)} labeled captions to {save_path}")
        return labeled_df
    
    def perform_eda_analysis(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple:
        """
        Perform comprehensive EDA analysis (unified for all datasets).
        
        Args:
            df (pd.DataFrame): Dataset with mood and caption columns
            dataset_name (str): Name of dataset for reporting
            
        Returns:
            Tuple: (emotion_counts, caption_lengths, word_counts)
        """
        print(f"\n EXPLORATORY DATA ANALYSIS - {dataset_name.upper()}")
        print("=" * 60)
        
        # 1. Dataset Overview
        print("1. DATASET OVERVIEW")
        print(f"   Total samples: {len(df):,}")
        print(f"   Number of unique emotions: {df['mood'].nunique()}")
        print(f"   Average caption length: {df['caption'].str.len().mean():.1f} characters")
        print(f"   Median caption length: {df['caption'].str.len().median():.1f} characters")
        
        # 2. Text Length Analysis
        caption_lengths = df['caption'].str.len()
        print(f"\n2. TEXT LENGTH STATISTICS")
        print(f"   Min length: {caption_lengths.min()} characters")
        print(f"   Max length: {caption_lengths.max()} characters")
        print(f"   25th percentile: {caption_lengths.quantile(0.25):.1f} characters")
        print(f"   75th percentile: {caption_lengths.quantile(0.75):.1f} characters")
        print(f"   Standard deviation: {caption_lengths.std():.1f} characters")
        
        # 3. Word count analysis
        word_counts = df['caption'].str.split().str.len()
        print(f"\n3. WORD COUNT STATISTICS")
        print(f"   Average words per caption: {word_counts.mean():.1f}")
        print(f"   Median words per caption: {word_counts.median():.1f}")
        print(f"   Min words: {word_counts.min()}")
        print(f"   Max words: {word_counts.max()}")
        
        # 4. Complete emotion distribution
        print(f"\n4. COMPLETE EMOTION DISTRIBUTION")
        emotion_counts = df['mood'].value_counts()
        total_samples = len(df)
        print("   Emotion (Count | Percentage)")
        print("   " + "-" * 35)
        for emotion, count in emotion_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {emotion:<12} ({count:>5} | {percentage:>5.1f}%)")
        
        # 5. Class imbalance analysis
        print(f"\n5. CLASS IMBALANCE ANALYSIS")
        most_common = emotion_counts.iloc[0]
        least_common = emotion_counts.iloc[-1]
        imbalance_ratio = most_common / least_common
        print(f"   Most common emotion: {emotion_counts.index[0]} ({most_common:,} samples)")
        print(f"   Least common emotion: {emotion_counts.index[-1]} ({least_common:,} samples)")
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        # 6. Sample examples for different emotions
        print(f"\n6. SAMPLE EXAMPLES BY EMOTION")
        print("   " + "-" * 50)
        sample_emotions = ['joy', 'sadness', 'anger', 'neutral', 'love', 'fear']
        for emotion in sample_emotions:
            if emotion in df['mood'].values:
                sample = df[df['mood'] == emotion]['caption'].iloc[0]
                print(f"   {emotion.upper()}: \"{sample[:80]}{'...' if len(sample) > 80 else ''}\"")
        
        print(f"\n7. DATA QUALITY INSIGHTS")
        print(f"   Empty captions: {df['caption'].isna().sum()}")
        print(f"   Very short captions (<10 chars): {(caption_lengths < 10).sum()}")
        print(f"   Very long captions (>200 chars): {(caption_lengths > 200).sum()}")
        print(f"   Unique captions: {df['caption'].nunique():,} ({(df['caption'].nunique()/len(df)*100):.1f}%)")
        
        print(f"\n EDA Complete! Dataset appears suitable for mood-conditioned generation.")
        
        return emotion_counts, caption_lengths, word_counts
    
    def balance_dataset_for_training(self, 
                                   df: pd.DataFrame, 
                                   max_samples_per_emotion: int = 1500, 
                                   min_samples_per_emotion: int = 100) -> pd.DataFrame:
        """
        Address class imbalance by limiting over-represented emotions and
        ensuring minimum representation for under-represented ones.
        
        Args:
            df (pd.DataFrame): DataFrame with mood and caption columns
            max_samples_per_emotion (int): Maximum samples per emotion
            min_samples_per_emotion (int): Minimum samples per emotion
            
        Returns:
            pd.DataFrame: Balanced dataset
        """
        print(f"\n ADDRESSING CLASS IMBALANCE")
        print("=" * 50)
        
        balanced_dfs = []
        emotion_counts = df['mood'].value_counts()
        
        for emotion in emotion_counts.index:
            emotion_data = df[df['mood'] == emotion]
            current_count = len(emotion_data)
            
            if current_count > max_samples_per_emotion:
                # Downsample over-represented emotions
                sampled_data = emotion_data.sample(n=max_samples_per_emotion, random_state=42)
                print(f"   {emotion}: {current_count} → {max_samples_per_emotion} (downsampled)")
            elif current_count < min_samples_per_emotion:
                # Upsample under-represented emotions (with replacement)
                sampled_data = emotion_data.sample(n=min_samples_per_emotion, replace=True, random_state=42)
                print(f"   {emotion}: {current_count} → {min_samples_per_emotion} (upsampled)")
            else:
                # Keep as is
                sampled_data = emotion_data
                print(f"   {emotion}: {current_count} (unchanged)")
            
            balanced_dfs.append(sampled_data)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n Balanced dataset size: {len(balanced_df)} (was {len(df)})")
        print("New emotion distribution:")
        print(balanced_df['mood'].value_counts().head(10))
        
        return balanced_df
    
    def create_training_dataset(self, 
                              df: pd.DataFrame, 
                              format_type: str = "complete_meme",
                              max_words: int = 15) -> Dataset:
        """
        Convert dataframe to HuggingFace Dataset with proper formatting.
        
        Args:
            df (pd.DataFrame): DataFrame with mood and caption columns
            format_type (str): "complete_meme", "structured", or "simple"
            max_words (int): Maximum words per caption for meme formatting
            
        Returns:
            Dataset: HuggingFace dataset ready for training
        """
        print(f"\n CREATING TRAINING DATASET")
        print("=" * 50)
        print(f"Format type: {format_type}")
        
        if format_type == "complete_meme":
            # Use complete output strategy (preserving full context)
            print("Applying intelligent text processing (preserving complete thoughts)...")
            
            # Clean the captions without aggressive truncation
            df["processed_caption"] = df["caption"].apply(
                lambda x: self._clean_text_for_memes(x, max_words)
            )
            
            # Filter out very short or empty captions
            df = df[df["processed_caption"].str.len() > 10].copy()
            print(f"After intelligent processing: {len(df)} complete captions remain")
            
            # Create training data using complete output strategy
            training_texts = []
            for _, row in df.iterrows():
                mood = row['mood']
                caption = row['processed_caption']
                
                # Complete output approach: Train on full, coherent captions
                training_prompt = f"Generate a {mood} meme caption: {caption}<|endoftext|>"
                training_texts.append(training_prompt)
            
            training_df = pd.DataFrame({"text": training_texts})
            
        elif format_type == "structured":
            # Create structured prompt format
            df["text"] = df.apply(
                lambda row: f"Generate a {row['mood']} caption: {row['caption']}<|endoftext|>", 
                axis=1
            )
            training_df = df[["text"]].copy()
            
        else:  # simple format
            df["text"] = df.apply(
                lambda row: f"{row['mood']}: {row['caption']}<|endoftext|>", 
                axis=1
            )
            training_df = df[["text"]].copy()
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(training_df)
        print(f" Created training dataset with {len(dataset)} examples")
        
        return dataset
    
    def _clean_text_for_memes(self, text: str, max_words: int = 15) -> str:
        """
        Clean and prepare text for meme-style captions without premature splitting.
        
        Args:
            text (str): Original text
            max_words (int): Maximum words per caption
            
        Returns:
            str: Cleaned meme-appropriate text
        """
        import re
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove quotes and special characters, keep basic punctuation and apostrophes
        text = re.sub(r'["""''`]', '', text)
        text = re.sub(r'[^\w\s,.!?\'-]', '', text)
        
        # Only limit length if significantly over limit, preserving complete thoughts
        words = text.split()
        if len(words) > max_words:
            # Find a good breaking point near the limit
            good_break_point = max_words
            for i in range(min(max_words, len(words) - 1), min(len(words), max_words + 3)):
                if i < len(words) and words[i].endswith(('.', '!', '?', ',')):
                    good_break_point = i + 1
                    break
            text = ' '.join(words[:good_break_point])
        
        return text.strip()
    
    def create_complete_meme_training_dataset(self, 
                                           df: pd.DataFrame, 
                                           max_words: int = 15) -> Dataset:
        """
        Create a complete meme training dataset using the complete output strategy.
        
        This function implements intelligent processing by:
        1. Smart Text Cleaning: Remove URLs, mentions, hashtags while preserving sentence structure
        2. Complete Thought Preservation: Allow full sentence generation (no premature cuts)
        3. Natural Length Limits: Only limit length when significantly over reasonable bounds
        4. Context-Aware Processing: Maintain emotional context and logical flow
        5. Flexible Training Format: Use prompts like "Generate a joy meme caption: {complete_thought}<|endoftext|>"
        
        Key Innovation: Post-Processing Intelligence
        Instead of forcing structure during training, we:
        - Train on complete outputs: Let GPT-2 generate coherent, flowing captions
        - Apply intelligent parsing later: Use intelligent_meme_parser() to format complete output
        - Respect natural language: Split at logical points (sentences, connectors, commas)
        - Maintain emotional integrity: Keep related emotional concepts together
        
        Args:
            df (pd.DataFrame): DataFrame with mood and caption columns
            max_words (int): Maximum words per caption for meme formatting
            
        Returns:
            Dataset: HuggingFace dataset ready for training with complete output strategy
        """
        print(f"\n CREATING COMPLETE MEME TRAINING DATASET")
        print("=" * 60)
        print("Using complete output strategy (preserving complete thoughts)...")
        
        # Clean the captions without aggressive truncation
        df = df.copy()
        df["processed_caption"] = df["caption"].apply(
            lambda x: self._clean_text_for_memes(x, max_words)
        )
        
        # Filter out very short or empty captions
        df = df[df["processed_caption"].str.len() > 10].copy()
        print(f"After intelligent processing: {len(df)} complete captions remain")
        
        # Create training data using complete output strategy
        training_texts = []
        for _, row in df.iterrows():
            mood = row['mood']
            caption = row['processed_caption']
            
            # Complete output approach: Train on full, coherent captions
            # No premature TOP/BOTTOM splitting - preserve complete thoughts
            training_prompt = f"Generate a {mood} meme caption: {caption}<|endoftext|>"
            training_texts.append(training_prompt)
        
        training_df = pd.DataFrame({"text": training_texts})
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(training_df)
        print(f" Created complete meme training dataset with {len(dataset)} examples")
        print(" Key features:")
        print("   - No early TOP/BOTTOM splitting")
        print("   - Preserves complete thoughts and emotional context")
        print("   - Allows for intelligent post-processing of generated output")
        print("   - Maintains sentence boundaries and logical flow")
        print("   - Better emotional alignment through preserved context")
        
        return dataset
    
    def tokenize_dataset(self, 
                        dataset: Dataset, 
                        model_name: str = "gpt2", 
                        max_length: int = 128) -> Tuple[Dataset, GPT2Tokenizer]:
        """
        Tokenize dataset for GPT-2 training.
        
        Args:
            dataset (Dataset): HuggingFace dataset to tokenize
            model_name (str): Model name for tokenizer
            max_length (int): Maximum sequence length
            
        Returns:
            Tuple[Dataset, GPT2Tokenizer]: Tokenized dataset and tokenizer
        """
        print(f"\n TOKENIZING DATASET")
        print("=" * 50)
        
        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length",
                            truncation=True, max_length=max_length)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        print(f" Tokenized {len(tokenized_dataset)} examples")
        
        return tokenized_dataset, tokenizer
    
    def train_multiple_models(self,
                            df: pd.DataFrame,
                            model_sizes: List[str] = ["base", "medium"],
                            output_base_dir: str = None,
                            dataset_name: str = "dataset",
                            epochs: int = 3,
                            **training_kwargs) -> Dict:
        """
        Train multiple model sizes with the same dataset for comparison.
        
        Args:
            df (pd.DataFrame): Balanced dataset
            model_sizes (List[str]): List of model sizes to train ("base", "medium")
            output_base_dir (str): Base directory for model outputs
            dataset_name (str): Dataset name for model naming
            epochs (int): Training epochs
            **training_kwargs: Additional training arguments
            
        Returns:
            Dict: Results from training multiple models
        """
        print(f"\n TRAINING MULTIPLE MODEL SIZES")
        print("=" * 60)
        
        if output_base_dir is None:
            output_base_dir = f"{self.project_root}/models"
        
        # Prepare dataset once
        training_dataset = self.create_complete_meme_training_dataset(df)
        tokenized_dataset, tokenizer = self.tokenize_dataset(training_dataset)
        
        results = {}
        
        for model_size in model_sizes:
            print(f"\n Training GPT-2 {model_size.upper()} model...")
            
            # Create model-specific output directory
            model_output_dir = f"{output_base_dir}/gpt2-{model_size}-{dataset_name}"
            
            try:
                # Import training function from caption_helpers
                from .caption_helpers import fine_tune_gpt2_enhanced
                
                trainer = fine_tune_gpt2_enhanced(
                    tokenized_dataset=tokenized_dataset,
                    tokenizer=tokenizer,
                    output_dir=model_output_dir,
                    model_size=model_size,
                    epochs=epochs,
                    **training_kwargs
                )
                
                results[model_size] = {
                    'trainer': trainer,
                    'output_dir': model_output_dir,
                    'status': 'completed'
                }
                
                print(f" {model_size.upper()} model training completed!")
                
            except Exception as e:
                print(f" {model_size.upper()} model training failed: {e}")
                results[model_size] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        print(f"\n TRAINING SUMMARY")
        print("=" * 40)
        for size, result in results.items():
            status = result['status']
            if status == 'completed':
                print(f"   {size.upper()}:  {result['output_dir']}")
            else:
                print(f"   {size.upper()}:  {result.get('error', 'Unknown error')}")
        
        return results


# Convenience function for easy usage
def create_dataset_factory(project_root: Optional[str] = None) -> UnifiedDatasetFactory:
    """
    Create a dataset factory instance.
    
    Args:
        project_root (str, optional): Root path of the project
        
    Returns:
        UnifiedDatasetFactory: Factory instance
    """
    return UnifiedDatasetFactory(project_root)


# Example usage and testing
if __name__ == "__main__":
    # Example: Switch between datasets
    factory = create_dataset_factory()
    
    # Load GoEmotions
    print("Testing GoEmotions loading...")
    goemotions_df = factory.load_dataset("goemotions")
    emotion_counts, _, _ = factory.perform_eda_analysis(goemotions_df, "GoEmotions")
    
    # Load ImgFlip (would take longer due to labeling)
    # print("Testing ImgFlip loading...")
    # imgflip_df = factory.load_dataset("imgflip", captions_to_process=1000)
    # factory.perform_eda_analysis(imgflip_df, "ImgFlip")




