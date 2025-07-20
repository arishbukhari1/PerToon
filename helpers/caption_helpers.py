import torch
from transformers import pipeline
from textblob import TextBlob
import re

def generate_meme_caption(mood, generator, model_type="fine_tuned", max_length=64, temperature=0.8, num_return_sequences=1):
    """
    Generate a meme caption for a given mood using the specified model.
    
    Args:
        mood (str): Target emotion (e.g., 'joy', 'sadness', 'anger')
        generator: Text generation pipeline
        model_type (str): 'fine_tuned' or 'baseline'
        max_length (int): Maximum token length for generation
        temperature (float): Sampling temperature (higher = more creative)
        num_return_sequences (int): Number of captions to generate
        
    Returns:
        dict: Generated caption with metadata
    """
    # Create appropriate prompt based on model type
    if model_type == "fine_tuned":
        # Use the training format our model expects
        prompt = f"Generate a {mood} meme:\nTOP:"
    else:
        # Simple prompt for baseline model
        if mood == "neutral":
            prompt = "Meme caption: "
        else:
            prompt = f"I feel {mood}. Meme caption: "
    
    try:
        # Generate text
        outputs = generator(
            prompt,
            max_new_tokens=max_length,
            #max_length=len(prompt.split()) + max_length,
            #num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=50256,
            eos_token_id=50256,
            repetition_penalty=1.2
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
    
    # Remove end tokens
    text = text.replace('<|endoftext|>', '').strip()
    
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