import pandas as pd
import ast
from typing import List, Dict, Optional

class KnowledgeBase:
    def __init__(self, ingredient_path='ingredientsList.csv'):
        # Load ingredients
        self.df = pd.read_csv(ingredient_path)
        
        # Create a lookup dictionary: Name (lowercase) -> Row Data
        self.ing_map = {}
        for _, row in self.df.iterrows():
            if pd.notna(row['name']):
                self.ing_map[row['name'].strip().lower()] = row.to_dict()
        
        # Pre-cache concerns for faster access
        self._concerns_cache = None
    
    def get_ingredient_info(self, name: str) -> Optional[Dict]:
        """Get details for a specific ingredient name."""
        return self.ing_map.get(str(name).strip().lower())
    
    def get_concerns(self) -> List[str]:
        """Extract all unique concerns from the 'who_is_it_good_for' column."""
        if self._concerns_cache is not None:
            return self._concerns_cache
            
        unique_concerns = set()
        for items in self.df['who_is_it_good_for']:
            try:
                # Parse string list "['Acne', 'Redness']" -> List
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
        """Check if an ingredient is good for a specific concern."""
        try:
            good_for = ast.literal_eval(ingredient_data.get('who_is_it_good_for', "[]"))
            # Case insensitive check
            return any(target_concern.lower() == item.strip().lower() for item in good_for)
        except:
            return False
    
    def check_skin_type_risk(self, ingredient_data: Dict, skin_type: str) -> bool:
        """Check if an ingredient should be avoided for a skin type."""
        try:
            avoid_list = ast.literal_eval(ingredient_data.get('who_should_avoid', "[]"))
            # Check if skin type is in the avoid list
            return any(skin_type.lower() == item.strip().lower() for item in avoid_list)
        except:
            return False
    
    def get_ingredients_for_concern(self, concern: str) -> List[str]:
        """Get all ingredients that address a specific concern."""
        ingredients = []
        for ing_name, ing_data in self.ing_map.items():
            if self.check_concern_match(ing_data, concern):
                ingredients.append(ing_name)
        return ingredients