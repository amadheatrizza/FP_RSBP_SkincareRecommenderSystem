import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import re

INPUT_FILE = "skincare_products_clean.csv"
OUTPUT_FILE = "skincare_products_with_images.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def scrape_lookfantastic_image(product_url):
    try:
        r = requests.get(product_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return og_image["content"]

        img = soup.find("img", {"itemprop": "image"})
        if img and img.get("src"):
            return img["src"]

    except Exception:
        pass

    return None

def scrape_bing_image(product_name):
    clean_name = re.sub(r"[^\w\s]", "", product_name)
    search_terms = " ".join(clean_name.split()[:7])
    query = f"{search_terms} official product packaging skincare"

    url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}"

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        results = soup.find_all("a", class_="iusc")

        for item in results:
            meta = item.get("m")
            if not meta:
                continue

            meta = re.sub(r"&quot;", '"', meta)
            match = re.search(r'"murl":"(.*?)"', meta)

            if match:
                image_url = match.group(1)
                if image_url.startswith("http"):
                    return image_url

    except Exception:
        pass

    return None

def main():
    print("📦 Loading CSV...")
    df = pd.read_csv(INPUT_FILE)

    if "image_url" not in df.columns:
        df["image_url"] = None

    total = len(df)
    print(f"🔍 Found {total} products")

    for index, row in df.iterrows():
        if pd.notna(row["image_url"]) and str(row["image_url"]).startswith("http"):
            continue

        product_name = str(row["product_name"])
        product_url = str(row["product_url"])

        image_url = None

        print(f"[{index+1}/{total}] 🔹 {product_name}")

        if product_url.startswith("http"):
            image_url = scrape_lookfantastic_image(product_url)

        if not image_url:
            image_url = scrape_bing_image(product_name)

        if image_url:
            df.at[index, "image_url"] = image_url
            print("   ✅ Image found")
        else:
            print("   ❌ Image not found")

        if index % 10 == 0:
            df.to_csv(OUTPUT_FILE, index=False)

        time.sleep(random.uniform(1.0, 1.8))

    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ DONE! Image scraping completed")


if __name__ == "__main__":
    main()
