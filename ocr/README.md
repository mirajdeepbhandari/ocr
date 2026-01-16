# License OCR API

FastAPI service for preprocessing and OCR recognition of license/ID card images.

## Features

- **Image Preprocessing Service**: Optimized for license/ID cards
  - Automatic resizing for performance
  - Denoising to remove image artifacts
  - Adaptive thresholding for better text contrast
  
- **OCR Service**: Powered by Surya models
  - High-accuracy text recognition
  - Confidence scores for each detection
  - Bounding box coordinates

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python main.py
```

Or with custom settings:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### 1. Preprocess and Recognize (Recommended)
**POST** `/ocr/preprocess-and-recognize`

Upload an image, preprocess it, and get OCR results.

```bash
curl -X POST "http://localhost:8000/ocr/preprocess-and-recognize" \
  -F "file=@path/to/image.png"
```

**Response:**
```json
{
  "success": true,
  "text_count": 5,
  "recognized_text": [
    {
      "text": "LICENSE NUMBER: 12345",
      "confidence": 0.95,
      "bbox": [10, 20, 100, 40]
    }
  ],
  "preprocessing_applied": true
}
```

### 2. Recognize Raw (No Preprocessing)
**POST** `/ocr/recognize-raw`

Perform OCR without preprocessing.

```bash
curl -X POST "http://localhost:8000/ocr/recognize-raw" \
  -F "file=@path/to/image.png"
```

### 3. Preprocess Only
**POST** `/ocr/preprocess-only`

Get preprocessed image as base64 (useful for debugging).

```bash
curl -X POST "http://localhost:8000/ocr/preprocess-only" \
  -F "file=@path/to/image.png"
```

**Response:**
```json
{
  "success": true,
  "preprocessed_image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "format": "base64_png"
}
```

### 4. Health Check
**GET** `/health`

Check service status.

## Python Client Example

```python
import requests

# Preprocess and recognize
with open('license.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/ocr/preprocess-and-recognize',
        files=files
    )
    result = response.json()
    
    for item in result['recognized_text']:
        print(f"{item['text']} (confidence: {item['confidence']:.2f})")
```

## Service Classes

### ImagePreprocessingService

Handles image preprocessing with configurable parameters:

```python
from main import ImagePreprocessingService
from PIL import Image

# Initialize service
service = ImagePreprocessingService(max_width=1200, denoise_strength=15)

# Preprocess image
image = Image.open('license.png')
preprocessed = service.preprocess(image)

# Or get as base64
base64_img = service.preprocess_to_base64(image)
```

### OCRService

Performs OCR using Surya models:

```python
from main import OCRService
from PIL import Image

# Initialize service
ocr = OCRService()

# Recognize text
images = [Image.open('license1.png'), Image.open('license2.png')]
results = ocr.recognize(images)

for result in results:
    for text_item in result:
        print(text_item['text'])
```

## Configuration

Edit the service initialization in `main.py`:

```python
# Adjust preprocessing parameters
preprocessing_service = ImagePreprocessingService(
    max_width=1200,         # Max width before resizing
    denoise_strength=15     # Denoising strength (10-30)
)
```

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
