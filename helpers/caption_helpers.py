from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from textblob import TextBlob
import re
import os

def load_fine_tuned_model(model_path=None, project_root=None, model_name="gpt2-mood-caption-v2", dataset_name=None):
    """
    Load a fine-tuned GPT-2 model with path flexibility for different environments.
    
    Args:
        model_path (str, optional): Direct path to the model. If None, constructs from project_root
        project_root (str, optional): Root path of the project. If None, uses "../" for local runs
        model_name (str): Name of the model directory
        dataset_name (str, optional): Dataset name for unified model naming (goemotions/imgflip)
        
    Returns:
        pipeline: Text generation pipeline with the loaded model
    """
    # Set default paths based on environment
    if project_root is None:
        project_root = ".."  # Default for local environment
    
    if model_path is None:
        # Support unified dataset-specific model names
        if dataset_name:
            model_path = f"{project_root}/models/gpt2-mood-caption-{dataset_name}-v1"
        else:
            model_path = f"{project_root}/models/{model_name}"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"  Model not found at {model_path}")
        print("Falling back to baseline GPT-2 model...")
        return pipeline("text-generation", model="gpt2", tokenizer="gpt2")
    
    try:
        # Load the fine-tuned model
        print(f" Loading fine-tuned model from: {model_path}")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Create pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print(f" Successfully loaded fine-tuned model!")
        return generator
        
    except Exception as e:
        print(f" Error loading fine-tuned model: {e}")
        print("Falling back to baseline GPT-2 model...")
        return pipeline("text-generation", model="gpt2", tokenizer="gpt2")

def generate_meme_caption(mood, generator, model_type="fine_tuned", max_length=40, temperature=0.95, num_return_sequences=1, diversity_mode="high", project_root=None):
    """
    Generate a meme caption for a given mood using the specified model.
    
    Args:
        mood (str): Target emotion (e.g., 'joy', 'sadness', 'anger')
        generator: Text generation pipeline
        model_type (str): 'fine_tuned' or 'baseline'
        max_length (int): Maximum token length for generation
        temperature (float): Sampling temperature (higher = more creative)
        num_return_sequences (int): Number of captions to generate
        diversity_mode (str): 'high', 'medium', or 'low' - controls diversity parameters
        project_root (str, optional): Root path of the project for file operations
        
    Returns:
        dict: Generated caption with metadata
    """
    # Dynamic few-shot examples based on mood category
    def get_diverse_examples(mood):
        """Get diverse examples based on mood to avoid bias"""
        positive_examples = [
            ("excitement", "WHEN THE WEEKEND FINALLY ARRIVES", "TIME TO PARTY"),
            ("joy", "FINALLY FINISHED MY HOMEWORK", "TIME FOR A NAP"),
            ("love", "WHEN YOU SEE YOUR CRUSH", "HEART GOES BRRR"),
            ("gratitude", "WHEN SOMEONE HELPS YOU", "FAITH IN HUMANITY RESTORED")
        ]
        
        negative_examples = [
            ("anger", "WHEN PEOPLE DON'T USE TURN SIGNALS", "MY BLOOD PRESSURE"),
            ("sadness", "WHEN YOU DROP YOUR ICE CREAM", "WHY CRUEL WORLD"),
            ("fear", "WHEN YOU HEAR FOOTSTEPS", "BEHIND YOU AT NIGHT"),
            ("disappointment", "WHEN THE RESTAURANT IS CLOSED", "BUT YOU DROVE 20 MINUTES")
        ]
        
        neutral_examples = [
            ("neutral", "WHEN YOU REALIZE IT'S MONDAY", "AGAIN"),
            ("confusion", "WHEN SOMEONE EXPLAINS CRYPTO", "STILL DON'T GET IT"),
            ("surprise", "WHEN YOU CHECK YOUR BANK ACCOUNT", "AFTER PAYDAY")
        ]
        
        # Choose example from different category to avoid bias
        if mood in ["joy", "love", "excitement", "gratitude", "pride", "relief"]:
            # For positive moods, use negative or neutral examples to increase diversity
            examples = negative_examples + neutral_examples
        elif mood in ["anger", "sadness", "fear", "disappointment", "grief", "disgust"]:
            # For negative moods, use positive or neutral examples
            examples = positive_examples + neutral_examples
        else:
            # For neutral moods, use any examples
            examples = positive_examples + negative_examples
        
        # Return a random example to add variation
        import random
        return random.choice(examples)
    
    # Set diversity parameters based on mode
    diversity_configs = {
        "high": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.1},
        "medium": {"temperature": 0.9, "top_p": 0.92, "top_k": 50, "repetition_penalty": 1.15}, 
        "low": {"temperature": 0.8, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.2}
    }
    
    config = diversity_configs.get(diversity_mode, diversity_configs["high"])
    
    # Create appropriate prompt based on model type
    if model_type == "fine_tuned":
        # Use dynamic few-shot prompting with diverse examples
        example_mood, example_top, example_bottom = get_diverse_examples(mood)
        prompt = f"""Generate a {example_mood} meme:
TOP: {example_top}
BOTTOM: {example_bottom}<|endoftext|>

Generate a {mood} meme:
TOP:"""
    else:
        # Simple prompt for baseline model
        if mood == "neutral":
            prompt = "Meme caption: "
        else:
            prompt = f"I feel {mood}. Meme caption: "
    
    try:
        # Generate text with improved diversity parameters
        outputs = generator(
            prompt,
            max_new_tokens=max_length,
            temperature=config["temperature"],  # Higher temperature for more creativity
            do_sample=True,
            top_p=config["top_p"],  # Higher top_p for more diverse sampling
            top_k=config["top_k"] if config["top_k"] > 0 else None,  # Top-k sampling for diversity
            pad_token_id=50256,
            eos_token_id=50256,
            repetition_penalty=config["repetition_penalty"],  # Lower repetition penalty
            typical_p=0.95,  # Add typical sampling for more natural diversity
            num_beams=1,  # Ensure we're using sampling, not beam search
        )
        
        results = []
        for output in outputs:
            # Extract generated text (remove prompt)
            generated_text = output["generated_text"].replace(prompt, "").strip()
            
            # Parse meme format if fine-tuned model
            if model_type == "fine_tuned":
                top_text, bottom_text = parse_meme_format(generated_text)
            else:
                # For baseline, treat as single caption
                top_text = generated_text.split('.')[0].strip().upper()  # Take first sentence
                bottom_text = ""
            
            # Calculate sentiment polarity
            full_caption = f"{top_text} {bottom_text}".strip()
            polarity = TextBlob(full_caption).sentiment.polarity
            
            results.append({
                "mood": mood,
                "top_text": top_text,
                "bottom_text": bottom_text,
                "full_caption": full_caption,
                "polarity": polarity,
                "model_type": model_type,
                "raw_output": generated_text
            })
        
        return results[0] if num_return_sequences == 1 else results
        
    except Exception as e:
        print(f"Error generating caption for mood '{mood}': {str(e)}")
        return {
            "mood": mood,
            "top_text": "ERROR",
            "bottom_text": "GENERATION FAILED",
            "full_caption": "ERROR: GENERATION FAILED",
            "polarity": 0.0,
            "model_type": model_type,
            "raw_output": str(e)
        }

def parse_meme_format(generated_text):
    """
    Parse generated text to extract TOP and BOTTOM meme components.
    
    Args:
        generated_text (str): Raw generated text from model
        
    Returns:
        tuple: (top_text, bottom_text)
    """
    # Clean up the text
    text = generated_text.strip()
    
    # --- NEW: Stop generation at the first <|endoftext|> token ---
    stop_token = "<|endoftext|>"
    if stop_token in text:
        text = text.split(stop_token)[0].strip()
    
    # Initialize defaults
    top_text = ""
    bottom_text = ""
    
    # Try to parse TOP: and BOTTOM: format first
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('TOP:'):
            top_text = line.replace('TOP:', '').strip()
        elif line.startswith('BOTTOM:'):
            bottom_text = line.replace('BOTTOM:', '').strip()
        elif not top_text and line:  # If no TOP: label found, use first non-empty line
            top_text = line
    
    # If we found a TOP: format but no BOTTOM:, try to find it in the remaining text
    if top_text and not bottom_text:
        # Look for BOTTOM: in the remaining text
        remaining_text = text.replace(top_text, '').strip()
        if 'BOTTOM:' in remaining_text:
            bottom_text = remaining_text.split('BOTTOM:')[-1].strip()
    
    # If we still don't have both parts, try to split the text
    if not top_text and text:
        # The text doesn't have explicit TOP:/BOTTOM: format, so we need to split it
        text_to_split = text
    elif top_text and not bottom_text:
        # We have top_text but no bottom_text, try to split the top_text itself
        text_to_split = top_text
        top_text = ""  # Reset to find both parts
    else:
        text_to_split = ""
    
    if text_to_split:
        # First try to split by sentence endings (., !, ?)
        import re
        sentences = re.split(r'[.!?]+', text_to_split)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have multiple sentences, use them
        if len(sentences) >= 2:
            top_text = sentences[0]
            bottom_text = sentences[1]
        elif len(sentences) == 1:
            # If only one sentence, try to split by comma
            if ',' in sentences[0]:
                parts = sentences[0].split(',', 1)  # Split on first comma only
                top_text = parts[0].strip()
                bottom_text = parts[1].strip()
            else:
                top_text = sentences[0]
        else:
            # If no clear sentences, try splitting by comma
            if ',' in text_to_split:
                parts = text_to_split.split(',', 1)  # Split on first comma only
                top_text = parts[0].strip()
                bottom_text = parts[1].strip()
            else:
                # Last resort: split by words
                words = text_to_split.split()
                if len(words) > 6:
                    # Take first half as TOP, second half as BOTTOM
                    mid_point = len(words) // 2
                    top_text = ' '.join(words[:mid_point])
                    bottom_text = ' '.join(words[mid_point:])
                else:
                    top_text = text_to_split
    
    # Clean and format
    top_text = top_text.strip().upper() if top_text else ""
    bottom_text = bottom_text.strip().upper() if bottom_text else ""
    
    return top_text, bottom_text

def display_meme_caption(caption_result, show_metadata=True):
    """
    Display a meme caption with top and bottom text only (no image box).
    
    Args:
        caption_result (dict): Caption generation result
        show_metadata (bool): Whether to show additional metadata
    """
    mood = caption_result['mood']
    top_text = caption_result['top_text']
    bottom_text = caption_result['bottom_text']
    polarity = caption_result['polarity']
    model_type = caption_result['model_type']
    
    print(f"MOOD: {mood.upper()}")
    print(f"MODEL: {model_type.replace('_', ' ').title()}")
    print()

    print(f"TOP TEXT:    {top_text if top_text else '(NO TOP TEXT)'}")
    print(f"BOTTOM TEXT: {bottom_text if bottom_text else '(NO BOTTOM TEXT)'}")

    if show_metadata:
        print(f"\nPolarity Score: {polarity:.3f}", end=" | ")
        if polarity > 0.1:
            print("Sentiment: Positive")
        elif polarity < -0.1:
            print("Sentiment: Negative")
        else:
            print("Sentiment: Neutral")
    
    print("-" * 50)

def generate_diverse_meme_captions(mood, generator, model_type="fine_tuned", num_variants=3, diversity_mode="high"):
    """
    Generate multiple diverse meme captions for the same mood to showcase improved diversity.
    
    Args:
        mood (str): Target emotion
        generator: Text generation pipeline
        model_type (str): 'fine_tuned' or 'baseline'
        num_variants (int): Number of diverse variants to generate
        diversity_mode (str): 'high', 'medium', or 'low'
        
    Returns:
        list: List of generated caption dictionaries
    """
    captions = []
    
    # Generate multiple variants with different strategies
    for i in range(num_variants):
        # Use different diversity modes for different variants
        if i == 0:
            current_mode = diversity_mode
        elif i == 1:
            # Second variant with higher temperature
            current_mode = "high" if diversity_mode != "high" else "medium"
        else:
            # Third variant with maximum diversity
            current_mode = "high"
        
        # Generate caption with unique seed
        import time
        time.sleep(0.01)  # Small delay to ensure different random seeds
        
        caption = generate_meme_caption(
            mood, 
            generator, 
            model_type=model_type, 
            diversity_mode=current_mode,
            temperature=0.9 + (i * 0.1)  # Gradually increase temperature
        )
        
        caption["variant_id"] = i + 1
        caption["diversity_mode"] = current_mode
        captions.append(caption)
    
    return captions

def analyze_diversity_metrics(captions):
    """
    Analyze diversity metrics across multiple generated captions.
    
    Args:
        captions (list): List of caption dictionaries
        
    Returns:
        dict: Diversity analysis results
    """
    if not captions:
        return {"error": "No captions provided"}
    
    # Extract texts for analysis
    top_texts = [c.get("top_text", "") for c in captions]
    bottom_texts = [c.get("bottom_text", "") for c in captions]
    full_captions = [c.get("full_caption", "") for c in captions]
    
    # Calculate uniqueness
    unique_top = len(set(top_texts))
    unique_bottom = len(set(bottom_texts))
    unique_full = len(set(full_captions))
    
    # Calculate word diversity
    all_words = []
    for caption in full_captions:
        words = caption.lower().split()
        all_words.extend(words)
    
    unique_words = len(set(all_words))
    total_words = len(all_words)
    word_diversity_ratio = unique_words / total_words if total_words > 0 else 0
    
    # Calculate polarity variance (emotional diversity)
    polarities = [c.get("polarity", 0) for c in captions]
    import statistics
    polarity_variance = statistics.variance(polarities) if len(polarities) > 1 else 0
    
    # Calculate average lengths
    avg_top_length = sum(len(text.split()) for text in top_texts) / len(top_texts)
    avg_bottom_length = sum(len(text.split()) for text in bottom_texts if text) / max(1, len([t for t in bottom_texts if t]))
    
    return {
        "total_captions": len(captions),
        "unique_top_texts": unique_top,
        "unique_bottom_texts": unique_bottom,
        "unique_full_captions": unique_full,
        "uniqueness_ratio": unique_full / len(captions),
        "word_diversity_ratio": word_diversity_ratio,
        "polarity_variance": polarity_variance,
        "avg_top_word_length": avg_top_length,
        "avg_bottom_word_length": avg_bottom_length,
        "polarities": polarities
    }

def display_diversity_comparison(mood, generator, model_type="fine_tuned"):
    """
    Display a comprehensive comparison showing improved diversity.
    
    Args:
        mood (str): Target emotion
        generator: Text generation pipeline
        model_type (str): Model type to test
    """
    print(f"DIVERSITY COMPARISON FOR MOOD: {mood.upper()}")
    print("=" * 60)
    
    # Generate diverse captions
    diverse_captions = generate_diverse_meme_captions(mood, generator, model_type, num_variants=5)
    
    # Display each variant
    for caption in diverse_captions:
        print(f"\nVARIANT {caption['variant_id']} (Mode: {caption['diversity_mode']}):")
        print(f"TOP:    {caption['top_text']}")
        print(f"BOTTOM: {caption['bottom_text']}")
        print(f"Polarity: {caption['polarity']:.3f}")
        print("-" * 40)
    
    # Analyze diversity
    diversity_metrics = analyze_diversity_metrics(diverse_captions)
    
    print(f"\nDIVERSITY ANALYSIS:")
    print("-" * 30)
    print(f"Unique full captions: {diversity_metrics['unique_full_captions']}/{diversity_metrics['total_captions']}")
    print(f"Uniqueness ratio: {diversity_metrics['uniqueness_ratio']:.2%}")
    print(f"Word diversity ratio: {diversity_metrics['word_diversity_ratio']:.2%}")
    print(f"Polarity variance: {diversity_metrics['polarity_variance']:.3f}")
    print(f"Average top text length: {diversity_metrics['avg_top_word_length']:.1f} words")
    print(f"Average bottom text length: {diversity_metrics['avg_bottom_word_length']:.1f} words")
    
    return diverse_captions, diversity_metrics

def intelligent_meme_parser(generated_text, mood=None):
    """
    Intelligently parse complete model output into TOP/BOTTOM meme format.
    
    This replaces brittle early splitting with intelligent post-processing
    that can handle complete, coherent thoughts from the model.
    
    Args:
        generated_text (str): Complete output from the fine-tuned model
        mood (str, optional): Original mood for context
        
    Returns:
        dict: Parsed meme with 'top', 'bottom', and metadata
    """
    # Clean the generated text
    text = generated_text.strip()
    
    # Remove any prompt artifacts
    if mood and f"Generate a {mood}" in text:
        text = re.sub(rf"Generate a {mood} meme caption:?\s*", "", text, flags=re.IGNORECASE)
    
    # Remove endoftext token if present
    text = text.replace("<|endoftext|>", "").strip()
    
    # Look for existing TOP/BOTTOM structure first
    top_match = re.search(r"TOP:\s*(.+?)(?:\s*BOTTOM:|$)", text, re.IGNORECASE | re.DOTALL)
    bottom_match = re.search(r"BOTTOM:\s*(.+?)$", text, re.IGNORECASE | re.DOTALL)
    
    if top_match:
        # Model already generated TOP/BOTTOM format
        top_text = top_match.group(1).strip()
        bottom_text = bottom_match.group(1).strip() if bottom_match else ""
    else:
        # Intelligently split coherent text into TOP/BOTTOM
        words = text.split()
        
        if len(words) <= 6:
            # Short text - use as TOP only
            top_text = text
            bottom_text = ""
        else:
            # Find natural break points for longer text
            sentence_breaks = []
            comma_breaks = []
            connector_breaks = []
            
            for i, word in enumerate(words):
                if word.endswith(('.', '!', '?')):
                    sentence_breaks.append(i + 1)
                elif word.endswith(','):
                    comma_breaks.append(i + 1)
                elif word.lower() in ['when', 'because', 'but', 'and', 'so', 'then', 'if']:
                    connector_breaks.append(i)
            
            # Choose best break point
            target_split = len(words) // 2
            
            # Prefer sentence breaks near the middle
            best_break = target_split
            if sentence_breaks:
                best_break = min(sentence_breaks, key=lambda x: abs(x - target_split))
            elif connector_breaks:
                best_break = min(connector_breaks, key=lambda x: abs(x - target_split))
            elif comma_breaks:
                best_break = min(comma_breaks, key=lambda x: abs(x - target_split))
            
            # Ensure reasonable split (not too unbalanced)
            if best_break < len(words) * 0.2:
                best_break = int(len(words) * 0.4)
            elif best_break > len(words) * 0.8:
                best_break = int(len(words) * 0.6)
            
            top_text = ' '.join(words[:best_break]).strip()
            bottom_text = ' '.join(words[best_break:]).strip()
    
    # Clean and format the text parts
    top_text = top_text.strip('.,!?').strip().upper()
    bottom_text = bottom_text.strip('.,!?').strip().upper()
    
    # Ensure reasonable length limits
    if len(top_text.split()) > 8:
        top_words = top_text.split()[:8]
        top_text = ' '.join(top_words)
    
    if len(bottom_text.split()) > 8:
        bottom_words = bottom_text.split()[:8]
        bottom_text = ' '.join(bottom_words)
    
    return {
        'top': top_text,
        'bottom': bottom_text,
        'original_text': generated_text,
        'parsing_method': 'existing_format' if top_match else 'intelligent_split',
        'mood': mood
    }
