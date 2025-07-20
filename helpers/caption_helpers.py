import torch
from transformers import pipeline
from textblob import TextBlob
import re

def generate_meme_caption(mood, generator, model_type="fine_tuned", max_length=64, temperature=0.8):
    """
    Generate a meme caption for a given mood using the specified model.
    """
    # Create appropriate prompt based on model type
    if model_type == "fine_tuned":
        prompt = f"Generate a {mood} meme:\\nTOP:"
    else:
        if mood == "neutral":
            prompt = "Meme caption: "
        else:
            prompt = f"I feel {mood}. Meme caption: "
    
    try:
        # Generate text
        outputs = generator(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=50256,
            eos_token_id=50256,
            repetition_penalty=1.2
        )
        
        output = outputs[0]
        generated_text = output["generated_text"].replace(prompt, "").strip()
        
        if model_type == "fine_tuned":
            top_text, bottom_text = parse_meme_format(generated_text)
        else:
            top_text = generated_text.split('.')[0].strip().upper()
            bottom_text = ""
        
        full_caption = f"{top_text} {bottom_text}".strip()
        polarity = TextBlob(full_caption).sentiment.polarity
        
        return {
            "mood": mood,
            "top_text": top_text,
            "bottom_text": bottom_text,
            "full_caption": full_caption,
            "polarity": polarity,
            "model_type": model_type,
            "raw_output": generated_text
        }
        
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
    """
    text = generated_text.strip().replace('<|endoftext|>', '').strip()
    top_text, bottom_text = "", ""

    lines = text.split('\\n')
    for line in lines:
        line = line.strip()
        if line.startswith('TOP:'):
            top_text = line.replace('TOP:', '').strip()
        elif line.startswith('BOTTOM:'):
            bottom_text = line.replace('BOTTOM:', '').strip()
        elif not top_text and line:
            top_text = line

    if not top_text and text:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            top_text, bottom_text = sentences[0], sentences[1]
        elif len(sentences) == 1:
            top_text = sentences[0]
        else:
            top_text = text

    return top_text.strip().upper(), bottom_text.strip().upper()

def display_meme_caption(caption_result, show_metadata=True):
    """
    Display a meme caption result in a formatted way.
    """
    print(f"MOOD: {caption_result['mood'].upper()}")
    print(f"MODEL: {caption_result['model_type'].replace('_', ' ').title()}")
    print(f"TOP TEXT:    {caption_result['top_text'] or '(NO TOP TEXT)'}")
    print(f"BOTTOM TEXT: {caption_result['bottom_text'] or '(NO BOTTOM TEXT)'}")

    if show_metadata:
        polarity = caption_result['polarity']
        sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
        print(f"\\nPolarity Score: {polarity:.3f} | Sentiment: {sentiment}")
    print("-" * 50)