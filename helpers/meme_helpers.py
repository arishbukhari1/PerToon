from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_meme_direct_pil(image_path, top_text="", bottom_text="", output_path="meme.jpg", size=(256, 256)):
    """
    Create meme directly with PIL for reliable text rendering.
    
    Args:
        image_path (str): Path to input image
        top_text (str): Text for top of meme
        bottom_text (str): Text for bottom of meme  
        output_path (str): Path for output image
        size (tuple): Output image size (width, height)
    
    Returns:
        PIL.Image: The created meme image
    """
    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Calculate font size based on image size
    font_size = int(width * 0.09)
    outline_width = max(2, font_size // 15)

    # Load font with proper fallbacks
    font = None
    font_paths = [
        "impact.ttf", "Impact.ttf", "IMPACT.TTF",
        "arial.ttf", "Arial.ttf", "arialbd.ttf",
        "/System/Library/Fonts/Impact.ttf",  # macOS
        "/Windows/Fonts/impact.ttf",  # Windows
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"  # Linux
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, size=font_size)
            print(f"Using font: {font_path}")
            break
        except IOError:
            continue

    if font is None:
        font = ImageFont.load_default()
        print("Using default font")

    def draw_meme_text(text, y_position):
        """Helper function to draw outlined text"""
        text = text.upper()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x = (width - text_width) / 2

        # Draw black outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y_position + dy), text, font=font, fill="black")

        # Draw white text
        draw.text((x, y_position), text, font=font, fill="white")

    def draw_wrapped_text(text, y_position, is_bottom=False):
        text = text.upper()
        
        # --- Remove trailing punctuation from bottom text ---
        if is_bottom and text.endswith(('.', ',', '!', '?')):
            text = text[:-1]

        # --- Automatically wrap text ---
        # Character width is roughly half the font size
        char_width_ratio = 1.8 
        chars_per_line = int(width / (font_size / char_width_ratio))
        lines = textwrap.wrap(text, width=chars_per_line)
        
        line_height = font.getbbox('A')[3] + font_size // 4

        # Adjust starting y_position for multi-line bottom text
        if is_bottom:
            y_position -= (len(lines) - 1) * line_height

        for line in lines:
            # Get text size for centering
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            x = (width - line_width) / 2
            
            # Draw outline
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y_position + dy), line, font=font, fill="black")
            
            # Draw main text
            draw.text((x, y_position), line, font=font, fill="white")
            y_position += line_height # Move to the next line

    # Draw top text
    if top_text:
        # draw_meme_text(top_text, font_size // 2)
        draw_wrapped_text(top_text, y_position=(font_size // 2))

    # Draw bottom text
    if bottom_text:
        # text_bbox = draw.textbbox((0, 0), bottom_text.upper(), font=font)
        text_bbox = draw.textbbox((0, 0), "A", font=font)
        text_height = text_bbox[3] - text_bbox[1]
        # draw_meme_text(bottom_text, height - text_height - font_size // 2)
        initial_y = height - text_height - (font_size // 2)
        draw_wrapped_text(bottom_text, y_position=initial_y, is_bottom=True)

    # Save with high quality
    img.save(output_path, "JPEG", quality=98, optimize=True)
    print(f"Meme created and saved: {output_path}")
    return img