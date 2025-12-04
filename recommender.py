import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import sparse

from knowledge_base import INGREDIENT_KNOWLEDGE, SENSITIVE_AVOID, get_explanation_template

class SkincareRecommender:
    def __init__(self, data_path, ml_model_type='ridge', ml_weight=0.4, random_state=42):
        self.data_path = data_path
        self.ml_model_type = ml_model_type
        self.ml_weight = float(ml_weight)
        self.random_state = random_state

        self.df = pd.read_csv(data_path, sep=';', encoding='latin-1')
        self._preprocess_data()
        self._build_hybrid_engine()
        try:
            self._train_ml_model()
            self.ml_ready = True
        except Exception as e:
            print("ML training failed:", e)
            self.ml_ready = False

    def _preprocess_data(self):
        def clean_price(price_str):
            if pd.isna(price_str): return 0.0
            clean_str = str(price_str).replace('Rp','').replace('.','').strip()
            if ',' in clean_str:
                clean_str = clean_str.split(',')[0]
            try: return float(clean_str)
            except: return 0.0

        self.df['price_cleaned'] = self.df['price'].apply(clean_price)

        skin_cols = ['Sensitive','Combination','Oily','Dry','Normal']
        for c in skin_cols:
            if c not in self.df.columns:
                self.df[c] = 0
        self.df[skin_cols] = self.df[skin_cols].fillna(0).astype(int)

        self.df['notable_effects'] = self.df['notable_effects'].fillna('')
        self.df['description'] = self.df['description'].fillna('')
        self.df['product_type'] = self.df['product_type'].fillna('Other')

        self.df['full_text'] = (self.df['notable_effects'] + ' ' + self.df['description']).str.lower()

    def _build_hybrid_engine(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['full_text'])

    def _compute_rule_score_for_row(self, row, skin_type, concern):
        score = 0
        reasons = []
        text_data = row['full_text']

        target_ingredients = []
        for key, ings in INGREDIENT_KNOWLEDGE.items():
            if key.lower() in concern.lower():
                target_ingredients.extend(ings)
        if not target_ingredients:
            target_ingredients = [concern.lower()]

        found = [ing for ing in target_ingredients if ing in text_data]
        if found:
            score += 10 * len(found)
            reasons.append(get_explanation_template(concern, found[0]))

        if skin_type == 'Sensitive':
            bad = [b for b in SENSITIVE_AVOID if b in text_data]
            if bad:
                score -= 50
                reasons.append(f"Warning: Contains {bad[0]} which may irritate sensitive skin.")
            else:
                score += 5
                reasons.append("Safe for Sensitive Skin.")

        if not reasons:
            reasons.append("Matches your skin profile.")

        return score, ' '.join(reasons)

    def _compute_rule_scores_dataset(self, default_concern='Hydration', default_skin='Normal'):
        scores = []
        explanations = []
        for _, row in self.df.iterrows():
            s, e = self._compute_rule_score_for_row(row, default_skin, default_concern)
            scores.append(s)
            explanations.append(e)
        self.df['rule_score'] = scores
        self.df['rule_explanation'] = explanations

    def _build_feature_matrix(self, df_subset=None):
        if df_subset is None:
            df_subset = self.df

        X_text = self.tfidf.transform(df_subset['full_text'])

        price = df_subset['price_cleaned'].values.reshape(-1,1)
        if not hasattr(self,'price_scaler'):
            self.price_scaler = StandardScaler()
            price_scaled = self.price_scaler.fit_transform(price)
        else:
            price_scaled = self.price_scaler.transform(price)

        skin_cols = ['Sensitive','Combination','Oily','Dry','Normal']
        skin_flags = df_subset[skin_cols].values.astype(float)

        X_num = sparse.csr_matrix(np.hstack([price_scaled, skin_flags]))
        X = sparse.hstack([X_text, X_num], format='csr')
        return X

    def _train_ml_model(self):
        self._compute_rule_scores_dataset()
        X = self._build_feature_matrix()
        y = self.df['rule_score'].values.astype(float)

        self.target_scaler = MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1,1)).ravel()

        # FORCE RANDOM FOREST
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X, y_scaled)
        self.ml_model = model

    def retrain_ml(self, ml_model_type=None, ml_weight=None):
        if ml_model_type:
            self.ml_model_type = ml_model_type
        if ml_weight is not None:
            self.ml_weight = float(ml_weight)
        self._train_ml_model()
        self.ml_ready = True

    def inference(self, skin_type, concern, product_type=None, min_price=0, max_price=10_000_000, top_k=5):
        mask = (self.df['price_cleaned'] >= min_price) & (self.df['price_cleaned'] <= max_price)

        if skin_type in ['Sensitive','Combination','Oily','Dry','Normal']:
            mask &= (self.df[skin_type] == 1)

        if product_type and product_type != 'All':
            mask &= (self.df['product_type'] == product_type)

        candidates = self.df[mask].copy()
        if candidates.empty:
            return pd.DataFrame()

        scores = []
        explanations = []
        for _, row in candidates.iterrows():
            s, e = self._compute_rule_score_for_row(row, skin_type, concern)
            scores.append(s)
            explanations.append(e)
        candidates['expert_score'] = scores
        candidates['explanation'] = explanations

        if self.ml_ready:
            Xcand = self._build_feature_matrix(candidates)
            pred_scaled = self.ml_model.predict(Xcand)

            try:
                pred_orig = self.target_scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
            except:
                pred_orig = pred_scaled * candidates['expert_score'].max()

            candidates['ml_score'] = pred_orig
            alpha = 1 - self.ml_weight
            candidates['final_score'] = alpha*candidates['expert_score'] + self.ml_weight*candidates['ml_score']
            candidates['explanation'] += " ML signal added."
        else:
            candidates['ml_score'] = 0
            candidates['final_score'] = candidates['expert_score']

        return candidates.sort_values('final_score', ascending=False).head(top_k)[[
            'product_name','brand','price_cleaned','product_type','description',
            'explanation','picture_src','expert_score','ml_score','final_score'
        ]]

    def get_product_types(self):
        return ['All'] + sorted(self.df['product_type'].unique())

    def debug_sample(self, n=5):
        return self.df[['product_name','rule_score','rule_explanation']].sort_values('rule_score', ascending=False).head(n)
