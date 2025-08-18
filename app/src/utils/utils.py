import re
import requests

def extract_links(text: str) -> list:
    # Regex for URLs (http, https, www)
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    return re.findall(url_pattern, text)

def fetch_url_content(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
