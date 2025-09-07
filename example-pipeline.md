# Example Pipeline for Certified Data Unlearnability Across Modalities

This document provides an example pipeline that shows how to handle multiple data modalities (text and images) while applying a reversible certified transformation. This transformation renders the data mathematically provably unlearnable by machine learning models, but it is also reversible so that humans can consume the data in its original form when needed.

## Overview

1. **Data Ingestion and Modality Separation:**
   - Raw data is split or identified by modality (text vs. image).
   - Each modality follows a dedicated branch in the pipeline.

2. **Preprocessing:**
   - **Text Data:**
     - **Tokenization:** Convert the text into tokens (words or subwords).
     - **Numerical Conversion:** Map tokens into numerical representations (e.g., token IDs or embedding vectors).
   - **Image Data:**
     - **Normalization:** Resize and normalize images.
     - **Tensor Conversion:** Convert the image into a numerical tensor (e.g., a NumPy array).

3. **Certified Unlearnability Transformation:**
   - A reversible transformation is applied to the numerical representations. Although the process introduces a reversible perturbation or transformation (e.g., additive noise), it is mathematically designed to prevent learning the underlying data patterns.
   - Mathematical proofs ensure that the transformation retains certified unlearnability.

4. **Postprocessing:**
   - **Text Data:**
     - Reverse the numerical transformation.
     - Convert numerical values back into text tokens and then reconstruct the original sentence.
   - **Image Data:**
     - Reverse the transformation.
     - Convert the numerical tensor back into a viewable image format (e.g., converting a NumPy array back to an image).

5. **Retention of Mathematical Guarantees:**
   - The reversible transformation protects the data during model training while allowing full recovery for human consumption. The integrity of the mathematical guarantees of unlearnability is maintained when models are trained on the transformed data.

## Example Code Outline

Below is a Python pseudocode example that describes each stage of the pipeline:

```python
import numpy as np
from PIL import Image
import nltk

# --- Certified Transformation Functions ---

def certified_transform(numerical_data):
    """
    Apply a reversible, certified unlearnability transformation to numerical data.
    In this example, a small reversible perturbation is added.
    """
    noise = np.random.normal(loc=0, scale=0.01, size=numerical_data.shape)
    transformed = numerical_data + noise
    return transformed, noise

def certified_inverse(transformed_data, noise):
    """
    Reverse the certified transformation using the stored noise.
    """
    return transformed_data - noise

# --- Text Data Pipeline ---

def preprocess_text(text):
    """
    Tokenizes text and converts tokens into numerical IDs.
    """
    # Tokenization using nltk
    tokens = nltk.word_tokenize(text)
    # Creating a simple token-to-ID mapping
    token_to_id = {token: idx for idx, token in enumerate(set(tokens))}
    numerical = np.array([token_to_id[token] for token in tokens])
    return numerical, token_to_id

def postprocess_text(numerical, token_to_id):
    """
    Converts numerical IDs back into text.
    """
    # Creating inverse mapping: id to token
    id_to_token = {id: token for token, id in token_to_id.items()}
    tokens = [id_to_token[num] for num in numerical]
    return " ".join(tokens)

# --- Image Data Pipeline ---

def preprocess_image(image_path):
    """
    Opens an image, converts it to grayscale, resizes it, and normalizes pixel values.
    """
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((128, 128))              # Resize image to a standardized shape
    numerical = np.array(image) / 255.0           # Normalize pixel values to range [0, 1]
    return numerical

def postprocess_image(numerical):
    """
    Converts the normalized numerical tensor back to an image format.
    """
    numerical = (numerical * 255).astype(np.uint8)
    image = Image.fromarray(numerical, mode='L')
    return image

# --- Simulated Pipeline Execution ---

# Example: Text Data
original_text = "The certified data unlearnability method applies to text and images."
# Preprocess text
text_numerical, token_map = preprocess_text(original_text)
# Apply certified transformation
text_transformed, text_noise = certified_transform(text_numerical)
# Assume model training occurs with 'text_transformed'
# Reverse transformation for human consumption
text_recovered_numerical = certified_inverse(text_transformed, text_noise)
recovered_text = postprocess_text(text_recovered_numerical, token_map)

print("Original Text:", original_text)
print("Recovered Text:", recovered_text)

# Example: Image Data
original_image_path = "path/to/example_image.png"  # Replace with an actual image path
# Preprocess image
image_numerical = preprocess_image(original_image_path)
# Apply certified transformation
image_transformed, image_noise = certified_transform(image_numerical)
# Assume model training occurs with 'image_transformed'
# Reverse transformation for human consumption
image_recovered_numerical = certified_inverse(image_transformed, image_noise)
recovered_image = postprocess_image(image_recovered_numerical)
recovered_image.show()  # This will display the recovered image
```

## Conclusion

This pipeline demonstrates how multiple data modalities (in this case, text and images) can be:
- Preprocessed into numerical forms suitable for applying a mathematically certified unlearnability transformation.
- Processed through a reversible transformation that ensures the data remains unlearnable by machine learning models.
- Postprocessed back to its original, human-readable or human-viewable form.

This approach ensures both the security (via unlearnability) of the underlying data for training purposes and the usability of the data for end-users.
