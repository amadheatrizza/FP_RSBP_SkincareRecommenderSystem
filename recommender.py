import pandas as pd
import numpy as np
from knowledge_base import analyze_product

class SkincareRecommender:
    def __init__(self, data_path):
        # Load Data
        self.df = pd.read_csv(data_path, sep=';', encoding='latin-1')
        self._preprocess()

    def _preprocess(self):
        # 1. Clean Price
        def clean_price(price_str):
            if pd.isna(price_str): return 0
            # Remove Rp, dots, whitespace
            s = str(price_str).replace('Rp', '').replace('.', '').strip()
            # Handle comma decimals
            if ',' in s: s = s.split(',')[0]
            try: return float(s)
            except: return 0
        
        self.df['price_cleaned'] = self.df['price'].apply(clean_price)
        
        # 2. Clean Text
        self.df['notable_effects'] = self.df['notable_effects'].fillna('')
        self.df['description'] = self.df['description'].fillna('')
        self.df['product_type'] = self.df['product_type'].fillna('Other')
        
        # Create a "Full Knowledge Context" string for scanning
        self.df['knowledge_context'] = (
            self.df['product_name'] + " " + 
            self.df['notable_effects'] + " " + 
            self.df['description']
        )
        
        # 3. Clean Binary Columns
        for col in ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def inference(self, skin_type, concern, product_type=None, max_price=10000000):
        """
        Runs the Rule-Based Inference.
        1. Hard Filters (Price, Category, Skin Type Label)
        2. Knowledge Scoring (Ingredients analysis)
        """
        
        # --- PHASE 1: HARD FILTERS (Pruning the Search Space) ---
        mask = (self.df['price_cleaned'] <= max_price)
        
        # Only keep products labeled for this skin type (if the label exists)
        if skin_type in ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']:
            mask = mask & (self.df[skin_type] == 1)
            
        if product_type and product_type != "All":
            mask = mask & (self.df['product_type'] == product_type)
            
        candidates = self.df[mask].copy()
        
        if candidates.empty:
            return pd.DataFrame()

        # --- PHASE 2: KNOWLEDGE REASONING (The "Thinking" Part) ---
        scores = []
        explanation_htmls = []
        
        for index, row in candidates.iterrows():
            # Call the Knowledge Base logic
            score, reasons = analyze_product(
                text=row['knowledge_context'],
                concern=concern,
                skin_type=skin_type
            )
            
            scores.append(score)
            # Format reasons into HTML for the UI
            explanation_htmls.append("<br>".join(reasons))

        candidates['final_score'] = scores
        candidates['explanation_html'] = explanation_htmls
        
        # --- PHASE 3: RANKING ---
        # Sort by score descending
        results = candidates.sort_values(by='final_score', ascending=False)
        
        # Filter out negative scores (Safety Violations) unless strictly necessary
        results = results[results['final_score'] > 0]
        
        return results.head(5)

    def get_product_types(self):
        return ['All'] + sorted(self.df['product_type'].unique().tolist())
    
    def get_concerns(self):
        # Return keys from Knowledge Base for the UI dropdown
        from knowledge_base import KNOWLEDGE_BASE
        return list(KNOWLEDGE_BASE.keys())