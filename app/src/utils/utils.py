import re

def extract_links(text: str) -> list:
    # Regex for URLs (http, https, www)
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    return re.findall(url_pattern, text)