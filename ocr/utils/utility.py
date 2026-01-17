import re

def verify_payment_ocr(ocr_text: str):
    # Clean OCR noise (HTML, confidence, math)
    text = re.sub(
        r"<math>.*?<\/math>|<\/?[^>]+>|\|\s*\(confidence=[0-9.]+\)",
        "",
        ocr_text,
        flags=re.DOTALL
    )

    # Normalize common OCR issues
    text = text.replace("AΜ", "AM").replace("ΑΜ", "AM")

    # REQUIRED patterns
    patterns = {
        "status": r"\b(SUCCESS|FAILED|PENDING)\b",
        "amount": r"\bAmount\s*\(NPR\)\s*([\d,.]+)|\bNPR\s*([\d,.]+)",
        "reference_code": r"\bReference\s*Code\s*\n?(\d{6,})",
        "date_time": r"\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4},?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)",
    }

    data = {}

    # Extract + strip required fields
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        value = next((g for g in match.groups() if g), None) if match else None
        data[key] = value.strip() if isinstance(value, str) else None

    # Merchant detection → exact canonical value
    merchant_pattern = r"TEAM\s+BHARIYA[\s\n]+INC[\s\n]+PVT\.?\s*LTD"
    data["merchant"] = (
        "TEAM BHARIYA INC PVT.LTD"
        if re.search(merchant_pattern, text, re.IGNORECASE)
        else None
    )

    # FINAL validation 
    is_valid = all([
        data["status"] is not None and data["status"].upper() == "SUCCESS",
        data["amount"] is not None,
        data["reference_code"] is not None,
        data["date_time"] is not None,
        data["merchant"] == "TEAM BHARIYA INC PVT.LTD",
    ])

    return is_valid, data


def sort_text_lines_reading_order(text_lines, y_tolerance: int = 15):
    """
    Sort OCR TextLine objects in reading order:
    Groups lines by vertical position (with tolerance), then sorts left-to-right.
    Handles multi-column layouts properly.
    """
    if not text_lines:
        return []
    
    # Extract positions
    lines_with_pos = []
    for line in text_lines:
        if hasattr(line, "bbox") and line.bbox:
            x_min, y_min, _, y_max = line.bbox
        else:
            # polygon = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [p[0] for p in line.polygon]
            ys = [p[1] for p in line.polygon]
            x_min = min(xs)
            y_min = min(ys)
            y_max = max(ys)
        
        y_center = (y_min + y_max) / 2
        lines_with_pos.append((line, x_min, y_center))
    
    # Sort by y_center first to process top-to-bottom
    lines_with_pos.sort(key=lambda x: x[2])
    
    # Group lines that are on the same horizontal level
    grouped_lines = []
    current_group = [lines_with_pos[0]]
    
    for i in range(1, len(lines_with_pos)):
        line, x_min, y_center = lines_with_pos[i]
        prev_line, prev_x, prev_y = current_group[-1]
        
        # If within y_tolerance, add to current group
        if abs(y_center - prev_y) <= y_tolerance:
            current_group.append(lines_with_pos[i])
        else:
            # Sort current group by x (left-to-right) and add to result
            current_group.sort(key=lambda x: x[1])
            grouped_lines.extend([item[0] for item in current_group])
            # Start new group
            current_group = [lines_with_pos[i]]
    
    # Add the last group
    current_group.sort(key=lambda x: x[1])
    grouped_lines.extend([item[0] for item in current_group])
    
    return grouped_lines



