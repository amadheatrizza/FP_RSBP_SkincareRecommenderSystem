import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import re

def scrape_bing():
    input_file = 'skincare_products_clean.csv'
    output_file = 'skincare_products_with_images.csv'
    
    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    if 'image_url' not in df.columns:
        df['image_url'] = None
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Found {len(df)} products. Starting Bing Search...")

    for index, row in df.iterrows():
        if pd.notna(row['image_url']) and str(row['image_url']).startswith('http'):
            continue

        # IMPROVED QUERY
        # Remove weird symbols from product name
        clean_name = re.sub(r'[^\w\s]', '', str(row['product_name']))
        # Limit to 6 words for search precision
        search_terms = " ".join(clean_name.split()[:6])
        
        # Query: "The Ordinary Niacinamide" + "official product packaging"
        query = f"{search_terms} skincare product official packaging white background".replace(' ', '+')
        url = f"https://www.bing.com/images/search?q={query}&first=1"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            img = soup.find('img', class_='mimg')
            
            if img and img.get('src'):
                df.at[index, 'image_url'] = img.get('src')
                print(f"[{index+1}/{len(df)}] ✅ Match: {search_terms}...")
            else:
                print(f"[{index+1}/{len(df)}] ❌ No match found")
                
        except Exception as e:
            print(f"[{index+1}] Error: {e}")
            
        time.sleep(random.uniform(0.5, 1.5))

        if index % 20 == 0:
            df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    scrape_bing()