import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import INGREDIENT_KNOWLEDGE, SENSITIVE_AVOID, get_explanation_template

class SkincareRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, sep=';', encoding='latin-1')
        self._preprocess_data()
        self._build_hybrid_engine()

    def _preprocess_data(self):
        # 1. Clean Price
        def clean_price(price_str):
            if pd.isna(price_str): return 0
            clean_str = str(price_str).replace('Rp', '').replace('.', '').strip()
            if ',' in clean_str: clean_str = clean_str.split(',')[0]
            try: return float(clean_str)
            except: return 0
        
        self.df['price_cleaned'] = self.df['price'].apply(clean_price)
        
        # 2. Fill NaNs
        skin_cols = ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']
        self.df[skin_cols] = self.df[skin_cols].fillna(0).astype(int)
        self.df['notable_effects'] = self.df['notable_effects'].fillna('')
        self.df['description'] = self.df['description'].fillna('')
        self.df['product_type'] = self.df['product_type'].fillna('Other')
        
        # 3. COMBINE Text for Searching
        # We search in both 'notable_effects' and 'description' to find ingredients
        self.df['full_text'] = (self.df['notable_effects'] + " " + self.df['description']).str.lower()

    def _build_hybrid_engine(self):
        # We keep TF-IDF as a "fallback" or "context" layer
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['notable_effects'])

    def inference(self, skin_type, concern, product_type=None, min_price=0, max_price=10000000):
        """
        The Core Inference Engine (Knowledge-Based + Rule-Based)
        """
        # --- PHASE 1: HARD CONSTRAINTS (Rules) ---
        mask = (self.df['price_cleaned'] >= min_price) & (self.df['price_cleaned'] <= max_price)
        
        if skin_type in ['Sensitive', 'Combination', 'Oily', 'Dry', 'Normal']:
            mask = mask & (self.df[skin_type] == 1)
            
        if product_type and product_type != "All":
            mask = mask & (self.df['product_type'] == product_type)
            
        candidates = self.df[mask].copy()
        
        if candidates.empty:
            return pd.DataFrame()

        # --- PHASE 2: KNOWLEDGE-BASED SCORING (The "Reasoning") ---
        
        # Get target ingredients for the user's concern
        # Default to 'Hydration' if concern not found in KB, or do keyword search
        target_ingredients = []
        for key, ingredients in INGREDIENT_KNOWLEDGE.items():
            if key.lower() in concern.lower():
                target_ingredients.extend(ingredients)
        
        # If no specific KB match, use the raw concern string for text search
        if not target_ingredients:
            target_ingredients = [concern.lower()]

        # Initialize Scoring and Explanation
        scores = []
        explanations = []

        for index, row in candidates.iterrows():
            score = 0
            reasons = []
            text_data = row['full_text']
            
            # RULE 1: Efficacy (Does it contain the ingredient?)
            found_ingredients = [ing for ing in target_ingredients if ing in text_data]
            if found_ingredients:
                score += 10 * len(found_ingredients)
                # Create Explanation
                top_ing = found_ingredients[0] # Pick the first one for brevity
                reasons.append(get_explanation_template(concern, top_ing))
            
            # RULE 2: Safety (Sensitive Skin Logic)
            if skin_type == 'Sensitive':
                bad_ingredients = [bad for bad in SENSITIVE_AVOID if bad in text_data]
                if bad_ingredients:
                    score -= 50 # Heavy penalty
                    reasons.append(f"Warning: Contains {bad_ingredients[0]} which might irritate sensitive skin.")
                else:
                    score += 5 # Bonus for being "safe"
                    reasons.append("Safe for Sensitive Skin (No common irritants detected).")

            # RULE 3: Fallback Similarity (TF-IDF)
            # We add a small fraction of the cosine similarity to break ties
            # (Simplified here for speed, usually calculated in bulk)
            
            if not reasons:
                reasons.append("Matches your skin type criteria.")

            scores.append(score)
            explanations.append(" ".join(reasons))

        candidates['expert_score'] = scores
        candidates['explanation'] = explanations

        # --- PHASE 3: RANKING ---
        # Sort by Expert Score first
        recommendations = candidates.sort_values(by='expert_score', ascending=False).head(5)
        
        return recommendations[['product_name', 'brand', 'price_cleaned', 'product_type', 'description', 'explanation', 'picture_src', 'expert_score']]

    def get_product_types(self):
        return ['All'] + sorted(self.df['product_type'].unique().tolist())  