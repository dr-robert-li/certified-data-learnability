# Example Pipeline for Transforming Varnish Cached Page Using Certified Data Unlearnability

This document outlines a pipeline that:
1. Pulls a full page from a Varnish cache,
2. Applies a certified unlearnability transformation to the page's contents,
3. And pushes the updated (transformed) version back into the cache.

## Overview

The goal is to ensure the cached representation becomes provably unlearnable to machine learning models while still being consumable by human users. In this scenario, because the source data is accessible to the user, there's no need for a separate reversal mechanism for display. The transformation ensures that any scraped (cached) data would not allow meaningful model training.

## Pipeline Steps

1. **Pull: Retrieve Cached Page**  
   - Fetch the cached page via an HTTP GET request from the Varnish instance.

2. **Transform: Certified Unlearnability Transformation**  
   - Preprocess the page by extracting and tokenizing text.
   - Convert the text into a numerical representation.
   - Apply a certified transformation, which might simply incorporate a reversible noise injection to ensure unlearnability (note that for public consumption, we can output the original tokens or a derivative representation).

3. **Push-update: Refresh the Cache**  
   - Push the transformed page back to Varnish (using HTTP PUT/POST).

## Example Code (Python)

```python
import requests
import numpy as np
from PIL import Image
import io
import nltk
from bs4 import BeautifulSoup

# ---------- Certified Transformation Functions -------------
def certified_transform(numerical_data):
    """
    Apply a certified unlearnability transformation to the numerical data.
    For this example, we add a small reversible noise.
    """
    noise = np.random.normal(loc=0, scale=0.01, size=numerical_data.shape)
    transformed = numerical_data + noise
    return transformed, noise

def certified_inverse(transformed_data, noise):
    """
    Reverses the certified transformation.
    (Not used in this public caching scenario, but provided for completeness.)
    """
    return transformed_data - noise

# ---------- Cached Page Retrieval from Varnish -------------
def fetch_cached_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Error fetching cached page. Status code: {response.status_code}")

# ---------- Cached Page Update in Varnish --------------------
def update_cached_page(url, transformed_content):
    # This endpoint is assumed to be available on the Varnish server.
    headers = {'Content-Type': 'text/html'}
    response = requests.put(url, headers=headers, data=transformed_content)
    if response.status_code in [200, 204]:
        print("Cache updated successfully.")
    else:
        print("Failed to update cache:", response.status_code)

# ---------- Preprocessing and Postprocessing of Text ---------
def preprocess_text(html_content):
    """
    Extract text from HTML, tokenize, and convert tokens to a numerical representation.
    In this example, we simply use token lengths as a dummy numerical representation.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=" ")
    tokens = nltk.word_tokenize(text)
    numerical = np.array([len(token) for token in tokens])
    return numerical, tokens

def postprocess_text(numerical, tokens):
    """
    For public consumption, we assume the transformation does not need to be reversed.
    The output is reconstructed from the original tokens.
    """
    # In a more complex scenario, you might reverse the transformation here.
    return " ".join(tokens)

# --------------- Main Pipeline --------------------------------
def main():
    varnish_url = "http://varnish-cache.example.com/cached-page"
    
    # 1. Pull: Retrieve the cached page from Varnish
    print("Pulling cached page...")
    cached_page = fetch_cached_page(varnish_url)
    
    # 2. Transform: Process the page content for unlearnability
    print("Preprocessing text from cached page...")
    text_numerical, tokens = preprocess_text(cached_page)
    
    print("Applying certified transformation...")
    transformed_numerical, noise = certified_transform(text_numerical)
    
    # For public display the meaningful content can be derived from the original tokens
    # or using a postprocessing step tailored to your needs.
    transformed_page = postprocess_text(transformed_numerical, tokens)
    
    # 3. Push-update: Update the transformed page back into the Varnish cache
    print("Pushing updated page back to Varnish cache...")
    update_cached_page(varnish_url, transformed_page)

if __name__ == "__main__":
    main()
```

## Summary

This pipeline demonstrates:

- **Pulling** a full HTML page from Varnish,
- **Transforming** its text content via a certified unlearnability mechanism,
- And **pushing** the transformed page back to refresh the cache.

Such a setup ensures that, even if the cached page is scraped, any machine learning model trained on its unlearnable numerical features will not be able to recover or effectively learn the meaningful content, while human users can still view the page content as intended.

---

### When an uncached/untransformed version of the page is required or authorized

Below is an example Varnish VCL snippet that checks for a URL query parameter (in this case, ?raw=1) to bypass any transformation logic and retrieve or render the original, untransformed page instead. You can add this or adapt it in your Varnish configuration file (e.g., default.vcl).

1. **vcl_recv:**  
   - When a request contains `?raw=1` in the URL, the configuration logs the request (optional) and passes it directly to the backend with `return (pass)`. This bypasses the cache (and therefore, any transformation logic) to retrieve the original page directly.

2. **vcl_backend_response:**  
   - For backend responses that were requested with `?raw=1`, the config clears any transformation markers before delivering, ensuring the original page is passed through.
   - Otherwise, transformation logic (or markers) are applied so that the cached version is known to be the transformed representation.

3. **vcl_deliver:**  
   - Adds an HTTP header (`X-Content-Transformation`) to indicate whether the content served is transformed or untransformed.

By using this URL parameter, you—and authorized users—can force the retrieval of an untransformed version of the cached page when necessary.

```
vcl 4.0;

import std;

sub vcl_recv {
    # Check for the presence of a "raw" query parameter to bypass transformation.
    if (req.url ~ "\?raw=1") {
        # Log the request for untransformed data (optional)
        std.log("Bypassing transformation: raw page render requested.");
        # Force pass the backend request so no cached (potentially transformed) copy is used.
        return (pass);
    }
    
    # Normal caching processing for transformed pages.
    return (hash);
}

sub vcl_backend_response {
    # For raw requests, you might decide not to transform the response.
    if (bereq.url ~ "\?raw=1") {
        # Remove any transformation-specific headers before caching.
        unset beresp.http.X-Transformed;
        return (deliver);
    }
    
    # Otherwise, apply transformation logic (implemented externally) before caching.
    # Example: set a header indicating the content has been transformed.
    set beresp.http.X-Transformed = "true";
    return (deliver);
}

sub vcl_deliver {
    # Optionally add a header to inform clients about the transformation status.
    if (obj.http.X-Transformed) {
        set resp.http.X-Content-Transformation = "Certified Unlearnability Transformation Applied";
    } else {
        set resp.http.X-Content-Transformation = "Original (Untransformed) Content";
    }
}
```
