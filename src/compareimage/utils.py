"""
Utility functions for image comparison and analysis.
"""

import difflib


def get_color_name(bgr_color):
    """
    Convert BGR color values to a human-readable color name.

    Parameters:
    - bgr_color: BGR color values (as a tuple or list)

    Returns:
    - String with a human-readable color name
    """
    r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]

    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Gray": (128, 128, 128),
        "Orange": (255, 165, 0),
        "Purple": (128, 0, 128),
        "Brown": (165, 42, 42),
        "Pink": (255, 192, 203),
        "Lime": (50, 205, 50),
        "Navy": (0, 0, 128),
        "Teal": (0, 128, 128)
    }

    min_distance = float('inf')
    nearest_color = "Unknown"

    for color_name, (rc, gc, bc) in colors.items():
        distance = (r - rc) ** 2 + (g - gc) ** 2 + (b - bc) ** 2
        if distance < min_distance:
            min_distance = distance
            nearest_color = color_name

    if abs(r - g) < 20 and abs(r - b) < 20 and abs(g - b) < 20:
        if r < 30:
            return "Black"
        elif r > 220:
            return "White"
        elif r < 85:
            return "Dark Gray"
        elif r < 170:
            return "Gray"
        else:
            return "Light Gray"

    return nearest_color


def extract_text_in_region(tesseract_data, x, y, w, h):
    """
    Extract text from OCR data that falls within the specified region.

    Parameters:
    - tesseract_data: OCR data from pytesseract
    - x, y, w, h: Region coordinates and dimensions

    Returns:
    - String containing the text found in the region
    """
    text = []
    n_boxes = len(tesseract_data['text'])

    for i in range(n_boxes):
        if int(tesseract_data['conf'][i]) < 0 or not tesseract_data['text'][i].strip():
            continue

        x1 = tesseract_data['left'][i]
        y1 = tesseract_data['top'][i]
        w1 = tesseract_data['width'][i]
        h1 = tesseract_data['height'][i]

        if (x1 >= x - 5 and y1 >= y - 5 and
                x1 + w1 <= x + w + 5 and y1 + h1 <= y + h + 5):
            text.append(tesseract_data['text'][i])

    return " ".join(text).strip()


def get_text_change_description(old_text, new_text):
    """
    Generate a human-readable description of how text has changed.

    Parameters:
    - old_text: Original text
    - new_text: Modified text

    Returns:
    - String describing the changes
    """
    if not old_text and new_text:
        return f"Text added: '{new_text}'"
    elif old_text and not new_text:
        return f"Text removed: '{old_text}'"
    else:
        d = difflib.Differ()
        diff = list(d.compare(old_text.split(), new_text.split()))

        # Create a more readable diff
        changes = []
        removed = []
        added = []

        for word in diff:
            if word.startswith('- '):
                removed.append(word[2:])
            elif word.startswith('+ '):
                added.append(word[2:])

        if removed and added:
            changes.append(f"Words removed: '{' '.join(removed)}'")
            changes.append(f"Words added: '{' '.join(added)}'")
            return "; ".join(changes)
        else:
            return f"Text changed from '{old_text}' to '{new_text}'"