import pandas as pd
import ast
from knowledge_base import KnowledgeBase

class SkincareRecommender:
    def __init__(self, product_path='skincare_products_with_images.csv', ingredient_path='ingredientsList.csv'):
        self.kb = KnowledgeBase(ingredient_path)
        
        # Load products (fallback to clean if images not scraped yet)
        try:
            self.df = pd.read_csv(product_path)
        except FileNotFoundError:
            self.df = pd.read_csv('skincare_products_clean.csv')

        self._preprocess()

    def _preprocess(self):
        # 1. Clean Price
        def clean_price(p):
            try:
                if pd.isna(p): return 0.0
                return float(str(p).replace('£', '').strip())
            except:
                return 0.0
        
        self.df['price_cleaned'] = self.df['price'].apply(clean_price)
        
        # 2. Fill NaNs
        self.df['product_type'] = self.df['product_type'].fillna('Other')
        self.df['clean_ingreds'] = self.df['clean_ingreds'].fillna("[]")
        if 'image_url' not in self.df.columns:
            self.df['image_url'] = None

    def get_product_types(self):
        return ['All'] + sorted(self.df['product_type'].astype(str).unique().tolist())

    def get_concerns(self):
        return self.kb.get_concerns()

    def inference(self, skin_type, selected_concerns, product_type, max_price):
        # 1. Filter Candidates
        mask = (self.df['price_cleaned'] <= max_price)
        if product_type != 'All':
            mask = mask & (self.df['product_type'] == product_type)
            
        candidates = self.df[mask].copy()
        
        scores = []
        explanations = []

        # 2. Score Candidates
        for index, row in candidates.iterrows():
            score = 0
            explanation_html = ""
            
            # Parse ingredient string
            try:
                ing_list = ast.literal_eval(row['clean_ingreds'])
            except:
                ing_list = []

            found_benefits = []
            found_risks = []
            
            # CHECK INGREDIENTS
            for ing in ing_list:
                info = self.kb.get_ingredient_info(ing)
                if info:
                    # A. EFFICACY: Loop through ALL selected concerns
                    # If an ingredient helps multiple concerns, it gets points for EACH.
                    for concern in selected_concerns:
                        if self.kb.check_concern_match(info, concern):
                            score += 10
                            desc = info.get("what_does_it_do", "Beneficial ingredient")

                            # Extract bullet points
                            lines = desc.split("\n")
                            bullets = [line[1:].strip() for line in lines if line.strip().startswith("-")]

                            if not bullets:
                                cleaned = desc.replace(f"{ing.lower()} offers benefits such as:", "").strip()
                                cleaned = cleaned.replace(f"{ing.title()} offers benefits such as:", "").strip()
                                
                                sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
                                bullets = sentences  # Each sentence becomes a bullet

                            # Build UL items
                            ul_html = "".join([f"<li>{b}</li>" for b in bullets])

                            # Final formatted output
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

                    # B. SAFETY: Check Skin Type Risks
                    if self.kb.check_skin_type_risk(info, skin_type):
                        score -= 20 # Penalty
                        if ing not in found_risks:
                            found_risks.append(f"<b>{ing.title()}</b> (Avoid for {skin_type})")

            # 3. Build Explainable Output
            if found_benefits:
                # Show top 3 benefits
                items_html = "".join([f"<li>{item}</li>" for item in found_benefits[:3]])
                explanation_html += f"<ul>{items_html}</ul>"
            else:
                explanation_html += "<p style='color:grey; margin:0;'>ℹ️ No specific actives found.</p>"

            if found_risks:
                items_html = ", ".join(found_risks[:2])
                explanation_html += f"<p style='color:#C62828; margin-top:5px;'>⚠️ <b>Risks:</b> Contains {items_html}</p>"

            # Sensitive Skin Bonus
            if skin_type == 'Sensitive' and not found_risks:
                score += 5
                explanation_html += "<p style='color:#1565C0; font-size:0.9em'>🛡️ <b>Safe:</b> No irritants found.</p>"

            scores.append(score)
            explanations.append(explanation_html)

        # 4. Normalize Score (Cap at 100%)
        # We allow score to be negative (bad match) or >100 internally, but cap for display
        candidates['raw_score'] = scores
        candidates['final_score'] = [max(0, min(100, s)) for s in scores]
        candidates['explanation_html'] = explanations

        # Return Top 5 Matches
        return candidates.sort_values(by='final_score', ascending=False).head(5)