import pandas as pd
import ast

class KnowledgeBase:
    def __init__(self, ingredient_path='ingredientsList.csv'):
        # Load ingredients
        self.df = pd.read_csv(ingredient_path)
        
        # Create a lookup dictionary: Name (lowercase) -> Row Data
        self.ing_map = {}
        for _, row in self.df.iterrows():
            if pd.notna(row['name']):
                self.ing_map[row['name'].strip().lower()] = row.to_dict()

    def get_ingredient_info(self, name):
        """Get details for a specific ingredient name."""
        return self.ing_map.get(str(name).strip().lower())

    def get_concerns(self):
        """Extract all unique concerns from the 'who_is_it_good_for' column."""
        unique_concerns = set()
        for items in self.df['who_is_it_good_for']:
            try:
                # Parse string list "['Acne', 'Redness']" -> List
                valid_list = ast.literal_eval(items)
                for i in valid_list:
                    if i.strip(): unique_concerns.add(i.strip())
            except:
                continue
        return sorted(list(unique_concerns))

    def check_concern_match(self, ingredient_data, target_concern):
        """Check if an ingredient is good for a specific concern."""
        try:
            good_for = ast.literal_eval(ingredient_data.get('who_is_it_good_for', "[]"))
            # Case insensitive check
            return any(target_concern.lower() == item.strip().lower() for item in good_for)
        except:
            return False

    def check_skin_type_risk(self, ingredient_data, skin_type):
        """Check if an ingredient should be avoided for a skin type."""
        try:
            avoid_list = ast.literal_eval(ingredient_data.get('who_should_avoid', "[]"))
            # Check if skin type is in the avoid list (e.g., 'Sensitive' in ['Sensitive', 'Oily'])
            return any(skin_type.lower() == item.strip().lower() for item in avoid_list)
        except:
            return False