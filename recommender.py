import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler
from knowledge_base import KnowledgeBase
import re

class SkincareRecommender:
    def __init__(self, product_path='skincare_products_with_images.csv', ingredient_path='ingredientsList.csv'):
        self.kb = KnowledgeBase(ingredient_path)
        
        try:
            self.df = pd.read_csv(product_path)
        except FileNotFoundError:
            self.df = pd.read_csv('skincare_products_clean.csv')
        
        self._preprocess()
        self._prepare_features()
        
    def _preprocess(self):
        def clean_price(p):
            try:
                if pd.isna(p):
                    return 0.0
                return float(str(p).replace('£', '').strip())
            except:
                return 0.0
        
        self.df['price_cleaned'] = self.df['price'].apply(clean_price)
        
        self.df['product_type'] = self.df['product_type'].fillna('Other')
        self.df['clean_ingreds'] = self.df['clean_ingreds'].fillna("[]")
        
        if 'image_url' not in self.df.columns:
            self.df['image_url'] = None
    
        self.df['ingredients_text'] = self.df['clean_ingreds'].apply(
            lambda x: self._process_ingredients(x)
        )
        
    def _process_ingredients(self, ing_str):
        """Process ingredients into a richer text representation"""
        try:
            ingredients = ast.literal_eval(ing_str)
            if not ingredients:
                return ""
            
            enhanced_text = []
            for ing in ingredients[:20]:
                ing_lower = ing.lower().strip()
                info = self.kb.get_ingredient_info(ing_lower)
                if info:
                    benefits = info.get('what_does_it_do', '')
                    if benefits:
                        key_terms = self._extract_key_terms(benefits)
                        enhanced_text.extend([ing_lower] * 2)  
                        enhanced_text.extend(key_terms)
                    else:
                        enhanced_text.extend([ing_lower] * 2)
                else:
                    enhanced_text.append(ing_lower)
            
            return ' '.join(enhanced_text)
        except:
            return ""
    
    def _extract_key_terms(self, text):
        beneficial_terms = {
            'hydrat', 'moistur', 'sooth', 'calm', 'repair', 'protect', 
            'brighten', 'firm', 'smooth', 'clear', 'reduce', 'improve',
            'anti', 'acne', 'aging', 'wrinkle', 'pigment', 'redness',
            'barrier', 'sensitive', 'oil', 'dry', 'normal', 'combination'
        }
        
        words = re.findall(r'\b[a-z][a-z]+\b', text.lower())
        return [w for w in words if any(term in w for term in beneficial_terms)]
    
    def _prepare_features(self):

        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2), 
            max_features=500,   
            min_df=2,           
            max_df=0.8,         
            sublinear_tf=True   
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['ingredients_text'])
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
    
    def get_product_types(self):
        return ['All'] + sorted(self.df['product_type'].astype(str).unique().tolist())
    
    def get_concerns(self):
        return self.kb.get_concerns()
    
    def _create_user_profile(self, selected_concerns, skin_type):

        profile_ingredients = []
        profile_terms = []
        
        for concern in selected_concerns:
            concern_ingredients = self.kb.get_ingredients_for_concern(concern)
            for ing in concern_ingredients[:10]:  # Limit to top 10 per concern
                profile_ingredients.append(ing)
                profile_terms.append(ing)
                
                info = self.kb.get_ingredient_info(ing)
                if info:
                    desc = info.get('what_does_it_do', '')
                    if desc:
                        key_terms = self._extract_key_terms(desc)
                        profile_terms.extend(key_terms)
        
        skin_type_terms = {
            'Dry': ['hydrat', 'moistur', 'barrier', 'dry', 'repair'],
            'Oily': ['oil', 'control', 'matte', 'sebum', 'clear'],
            'Sensitive': ['sooth', 'calm', 'gentle', 'sensitive', 'irritat'],
            'Combination': ['balance', 'combination', 'tzone', 'adjust'],
            'Normal': ['maintain', 'healthy', 'normal', 'protect']
        }
        
        if skin_type in skin_type_terms:
            profile_terms.extend(skin_type_terms[skin_type])
        
        query_text = ' '.join(profile_terms)
        
        query_vector = self.tfidf_vectorizer.transform([query_text])
        
        return query_vector, query_text
    
    def _calculate_similarity_scores(self, query_vector):

        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        scores = cosine_similarities * 100
        
        scaler = MinMaxScaler(feature_range=(40, 100))
        scores_scaled = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        
        return scores_scaled
    
    def _apply_skin_type_adjustments(self, scores, skin_type):
        adjusted_scores = scores.copy()
        
        for idx, row in self.df.iterrows():
            product_idx = idx
            
            try:
                if isinstance(row['clean_ingreds'], str):
                    ingredients = ast.literal_eval(row['clean_ingreds'])
                else:
                    ingredients = []
            except:
                ingredients = []
            
            avoid_count = 0
            for ing in ingredients:
                info = self.kb.get_ingredient_info(ing)
                if info and self.kb.check_skin_type_risk(info, skin_type):
                    avoid_count += 1
            
            if avoid_count > 0:
                penalty = min(30, avoid_count * 8)  # Reduced penalty
                adjusted_scores[product_idx] -= penalty
            
            if skin_type == 'Sensitive' and avoid_count == 0:
                adjusted_scores[product_idx] += 10
        
        return np.clip(adjusted_scores, 0, 100)
    
    def _generate_explanation_html(self, product_idx, skin_type, selected_concerns, score):
        explanations = []
        
        try:
            ingredients = ast.literal_eval(self.df.iloc[product_idx]['clean_ingreds'])
        except:
            ingredients = []
        
        found_benefits = []
        found_risks = []
        
        for ing in ingredients:
            info = self.kb.get_ingredient_info(ing)
            if info:
                for concern in selected_concerns:
                    if self.kb.check_concern_match(info, concern):
                        desc = info.get("what_does_it_do", "Beneficial ingredient")
                        
                        lines = desc.split("\n")
                        bullets = [line[1:].strip() for line in lines if line.strip().startswith("-")]
                        
                        if not bullets:
                            cleaned = desc.replace(f"{ing.lower()} offers benefits such as:", "").strip()
                            cleaned = cleaned.replace(f"{ing.title()} offers benefits such as:", "").strip()
                            
                            sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
                            bullets = sentences  
                        
                        if bullets:
                            ul_html = "".join([f"<li>{b}</li>" for b in bullets])
                           
                            entry = f"""
                            <div style="margin-bottom:10px;">
                            <p style="margin:0;"><b>{ing.title()}</b> offers benefits such as:</p>
                            <ul style="margin-top:4px;">
                                {ul_html}
                            </ul>
                            </div>
                            """
                            
                            if entry not in found_benefits:
                                found_benefits.append(entry)
        
                if self.kb.check_skin_type_risk(info, skin_type):
                    if ing not in found_risks:
                        found_risks.append(f"<b>{ing.title()}</b> (Avoid for {skin_type})")
        
        if found_benefits:
            items_html = "".join([f"<li>{item}</li>" for item in found_benefits[:3]])
            explanation_html = f"<ul>{items_html}</ul>"
        else:
            explanation_html = "<p style='color:grey; margin:0;'>ℹ️ No specific actives found.</p>"
        
        if found_risks:
            items_html = ", ".join(found_risks[:2])
            explanation_html += f"<p style='color:#C62828; margin-top:5px;'>⚠️ <b>Risks:</b> Contains {items_html}</p>"
        
        if skin_type == 'Sensitive' and not found_risks:
            explanation_html += "<p style='color:#1565C0; font-size:0.9em'>🛡️ <b>Safe:</b> No irritants found.</p>"
        
        return explanation_html
    
    def inference(self, skin_type, selected_concerns, product_type, max_price):
        mask = (self.df['price_cleaned'] <= max_price)
        if product_type != 'All':
            mask = mask & (self.df['product_type'] == product_type)
        
        candidates = self.df[mask].copy()
        
        if candidates.empty:
            return pd.DataFrame()
        
        query_vector, _ = self._create_user_profile(selected_concerns, skin_type)
        
        all_scores = self._calculate_similarity_scores(query_vector)
        
        adjusted_scores = self._apply_skin_type_adjustments(all_scores, skin_type)
        
        results = []
        explanations = []
        candidate_indices = candidates.index.tolist()
        
        for idx in candidate_indices:
            score = adjusted_scores[idx]
            
            explanation = self._generate_explanation_html(
                idx, skin_type, selected_concerns, score
            )
            
            results.append(score)
            explanations.append(explanation)
        
        candidates['raw_score'] = results
        candidates['final_score'] = [max(0, min(100, s)) for s in results]
        candidates['explanation_html'] = explanations
        
        candidates = candidates.sort_values('final_score', ascending=False)
        
        return candidates[candidates['final_score'] >= 40].head(5)