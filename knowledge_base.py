# --- KNOWLEDGE ONTOLOGY ---
KNOWLEDGE_BASE = {
    "Acne": {
        "actives": [
            "salicylic acid", "benzoyl peroxide", "tea tree", "retinol", 
            "niacinamide", "sulfur", "clay", "charcoal", "kaolin", 
            "bentonite", "aha", "bha", "succinic acid", "centella"
        ],
        "explanation": "Contains {}, which targets acne bacteria, unclogs pores, and reduces inflammation."
    },
    "Anti-Aging": {
        "actives": [
            "retinol", "retinyl", "bakuchiol", "peptide", "ceramide", 
            "vitamin c", "hyaluronic acid", "collagen", "glycolic acid", 
            "lactic acid", "adenosine", "ginseng", "resveratrol", "q10"
        ],
        "explanation": "Contains {}, proven to stimulate collagen, reduce fine lines, and improve skin elasticity."
    },
    "Brightening": {
        "actives": [
            "vitamin c", "niacinamide", "alpha arbutin", "kojic acid", 
            "licorice", "azelaic acid", "tranexamic acid", "glycolic acid", 
            "glutathione", "galactomyces", "rice extract", "papaya"
        ],
        "explanation": "Contains {}, which inhibits melanin production to fade dark spots and brighten skin tone."
    },
    "Hydration": {
        "actives": [
            "hyaluronic acid", "sodium hyaluronate", "glycerin", "ceramide", 
            "squalane", "panthenol", "vitamin b5", "aloe vera", 
            "snail mucin", "betaine", "allantoin", "polyglutamic acid"
        ],
        "explanation": "Contains {}, a humectant/emollient that draws moisture into the skin and repairs the barrier."
    },
    "Redness & Soothing": {
        "actives": [
            "centella", "cica", "mugwort", "aloe vera", "green tea", 
            "chamomile", "calendula", "oat", "panthenol", "allantoin", 
            "heartleaf", "propolis"
        ],
        "explanation": "Contains {}, known for its anti-inflammatory properties to calm redness and irritation."
    },
    "Pore Care": {
        "actives": [
            "niacinamide", "salicylic acid", "bha", "witch hazel", 
            "clay", "charcoal", "green tea"
        ],
        "explanation": "Contains {}, which regulates oil production and tightens the appearance of pores."
    }
}

#CONSTRAINT RULES (SAFETY)
SAFETY_RULES = {
    "Sensitive": {
        "avoid": [
            "alcohol denat", "ethanol", "fragrance", "parfum", 
            "essential oil", "sodium lauryl sulfate", "sls", "menthol"
        ],
        "penalty_msg": "⚠️ Risk: Contains {}, a potential irritant for sensitive skin."
    },
    "Dry": {
        "avoid": ["alcohol denat", "ethanol", "clay", "charcoal"],
        "penalty_msg": "⚠️ Caution: Contains {}, which can be drying."
    },
    "Oily": {
        "avoid": ["mineral oil", "coconut oil", "shea butter", "beeswax", "lanolin"],
        "penalty_msg": "⚠️ Note: Contains {}, which is potentially heavy/comedogenic for oily skin."
    }
}

def analyze_product(text, concern, skin_type):
    text_lower = text.lower()
    score = 0
    reasons = []

    # 1. CHECK EFFICACY (Forward Chaining)
    # Does the product contain ingredients that solve the user's concern?
    
    if concern in KNOWLEDGE_BASE:
        knowledge = KNOWLEDGE_BASE[concern]
        found_actives = [ing for ing in knowledge['actives'] if ing in text_lower]
        
        if found_actives:
            # Found primary solution
            top_active = found_actives[0] # Take the first one found
            reasons.append(f"✅ {knowledge['explanation'].format(top_active.title())}")
            score += 10 + (len(found_actives) * 2) # Base 10 + bonus for multiple actives
        else:
            # Fallback: No specific active found
            score += 1
            reasons.append("ℹ️ Matches product category, but key active ingredients weren't explicitly listed.")
    else:
        # Fallback for custom text inputs
        reasons.append("ℹ️ General product recommendation.")

    # 2. CHECK SAFETY (Constraint Satisfaction)
    # Does the product contain things the user MUST avoid?
    if skin_type in SAFETY_RULES:
        rule = SAFETY_RULES[skin_type]
        found_bad = [bad for bad in rule['avoid'] if bad in text_lower]
        
        if found_bad:
            # VIOLATION FOUND
            bad_ing = found_bad[0]
            reasons.append(rule['penalty_msg'].format(bad_ing.title()))
            score -= 30 # Massive penalty
        else:
            if skin_type == 'Sensitive':
                reasons.append("🛡️ Safe: Free from common irritants (Alcohol/Fragrance).")
                score += 5

    return score, reasons