import pandas as pd
import ast
from typing import List, Dict, Optional

class KnowledgeBase:
    def __init__(self, ingredient_path='ingredientsList.csv'):
        self.df = pd.read_csv(ingredient_path)
       
        self.ing_map = {}
        for _, row in self.df.iterrows():
            if pd.notna(row['name']):
                self.ing_map[row['name'].strip().lower()] = row.to_dict()
        
        self._concerns_cache = None
    
    def get_ingredient_info(self, name: str) -> Optional[Dict]:
        return self.ing_map.get(str(name).strip().lower())
    
    def get_concerns(self) -> List[str]:
        if self._concerns_cache is not None:
            return self._concerns_cache
            
        unique_concerns = set()
        for items in self.df['who_is_it_good_for']:
            try:
                valid_list = ast.literal_eval(items)
                for i in valid_list:
                    cleaned = i.strip()
                    if cleaned and cleaned != ' ':
                        unique_concerns.add(cleaned)
            except:
                continue
        
        self._concerns_cache = sorted(list(unique_concerns))
        return self._concerns_cache
    
    def check_concern_match(self, ingredient_data: Dict, target_concern: str) -> bool:
        try:
            good_for = ast.literal_eval(ingredient_data.get('who_is_it_good_for', "[]"))
            return any(target_concern.lower() == item.strip().lower() for item in good_for)
        except:
            return False
    
    def check_skin_type_risk(self, ingredient_data: Dict, skin_type: str) -> bool:
        try:
            avoid_list = ast.literal_eval(ingredient_data.get('who_should_avoid', "[]"))
            return any(skin_type.lower() == item.strip().lower() for item in avoid_list)
        except:
            return False
    
    def get_ingredients_for_concern(self, concern: str) -> List[str]:
        ingredients = []
        for ing_name, ing_data in self.ing_map.items():
            if self.check_concern_match(ing_data, concern):
                ingredients.append(ing_name)
        return ingredients