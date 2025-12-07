import streamlit as st
import pandas as pd
from recommender import SkincareRecommender

st.set_page_config(page_title="AI Skincare Expert", layout="wide")

# CSS
st.markdown("""
<style>
    .product-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        color: #333;
    }
    .product-img {
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 8px;
    }
    .price-tag {
        background-color: #f0f0f0;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        color: #333;
    }
    .match-score {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    return SkincareRecommender()

recommender = load_recommender()

# --- SIDEBAR ---
st.sidebar.header("Skin Profile")
name = st.sidebar.text_input("Name", "User")
skin_type = st.sidebar.selectbox("Skin Type", ['Dry', 'Oily', 'Combination', 'Sensitive', 'Normal'])

# CHANGED: Multiselect for concerns
all_concerns = recommender.get_concerns()
selected_concerns = st.sidebar.multiselect("Skin Concerns (Select all that apply)", all_concerns, default=[all_concerns[0]])

st.sidebar.markdown("---")
st.sidebar.header("Filters")
p_type = st.sidebar.selectbox("Product Type", recommender.get_product_types())
max_price = st.sidebar.slider("Max Price (£)", 5.0, 100.0, 30.0)

# --- HOW IT WORKS EXPANDER ---
with st.sidebar.expander("ℹ️ How the Score Works"):
    st.write("""
    **The Percentage Match is calculated based on ingredients:**
    * **+10%** for every ingredient that treats your specific concern.
    * **+5%** bonus if the product is safe for sensitive skin (and you are sensitive).
    * **-20%** penalty if it contains ingredients bad for your skin type.
    * The score is capped at **100%**.
    """)

# --- MAIN ---
st.title("AI Skincare Recommender")
st.markdown(f"Finding the best **{p_type}** for **{skin_type}** skin.")

if st.sidebar.button("Get Recommendations"):
    if not selected_concerns:
        st.error("Please select at least one concern.")
    else:
        with st.spinner("Analyzing ingredients..."):
            # Pass list of concerns
            results = recommender.inference(skin_type, selected_concerns, p_type, max_price)
        
        if not results.empty:
            for _, row in results.iterrows():
                with st.container():
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    c1, c2 = st.columns([1, 4])
                    
                    with c1:
                        img_url = row['image_url'] if pd.notna(row['image_url']) else "https://via.placeholder.com/150?text=No+Image"
                        st.markdown(f'<img src="{img_url}" class="product-img">', unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown(f"### {row['product_name']} <span class='match-score'>{int(row['final_score'])}% Match</span>", unsafe_allow_html=True)
                        st.markdown(f"<span class='price-tag'>£{row['price_cleaned']:.2f}</span>", unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown(row['explanation_html'], unsafe_allow_html=True)
                        
                        with st.expander("Full Ingredient List"):
                            ing_clean = str(row['clean_ingreds']).replace('[','').replace(']','').replace("'", "")
                            st.caption(ing_clean)
                            
                        st.link_button("Buy Product", row['product_url'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No matches found. Try increasing your budget or changing the product type.")
else:
    st.info("👈 Please set your preferences on the left to start.")