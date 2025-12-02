# knowledge_base.py

# FACT: Mapping Ingredients to Skin Concerns
INGREDIENT_KNOWLEDGE = {
    "Acne": [
        "salicylic acid", "tea tree", "centella", "mugwort", 
        "niacinamide", "bha", "aha", "retinol", "zinc"
    ],
    "Brightening": [
        "vitamin c", "niacinamide", "alpha arbutin", "licorice", 
        "rice extract", "galactomyces", "tranexamic acid"
    ],
    "Anti-Aging": [
        "retinol", "peptide", "collagen", "ginseng", 
        "adenosine", "snail mucin", "ceramide"
    ],
    "Hydration": [
        "hyaluronic acid", "glycerin", "ceramide", "aloe vera", 
        "panthenol", "snail mucin", "betaine"
    ],
    "Soothing": [
        "centella", "aloe vera", "mugwort", "calendula", 
        "chamomile", "green tea", "panthenol"
    ]
}

# RULE: Ingredients to AVOID for Sensitive Skin
SENSITIVE_AVOID = [
    "alcohol", "parfum", "fragrance", "essential oil", "sls", "sulfate"
]

def get_explanation_template(concern, ingredient):
    return f"contains **{ingredient.title()}**, which is a proven ingredient for treating **{concern}**."