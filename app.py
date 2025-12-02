import streamlit as st
import pandas as pd
from recommender import SkincareRecommender

# --- CONFIG ---
st.set_page_config(page_title="GlowUp Expert System", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .stAlert { padding: 10px; border-radius: 10px; }
    .score-badge { 
        background-color: #4CAF50; color: white; padding: 5px 10px; 
        border-radius: 15px; font-weight: bold; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- INIT ---
@st.cache_resource
def load_engine():
    return SkincareRecommender('export_skincare.csv')

engine = load_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🧬 User Diagnostics")
    name = st.text_input("Name", "User")
    
    st.subheader("1. Skin Profile")
    skin_type = st.selectbox("Skin Type", ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
    
    st.subheader("2. Concerns")
    # We map the UI options to our Knowledge Base keys for better accuracy
    concern_options = ["Acne", "Brightening", "Anti-Aging", "Hydration", "Soothing", "General Care"]
    concern = st.selectbox("Primary Concern", concern_options)
    
    st.subheader("3. Preferences")
    p_type = st.selectbox("Product Type", engine.get_product_types())
    max_price = st.slider("Max Price (IDR)", 50000, 2000000, 500000)

# --- MAIN ---
st.title("🧬 GlowUp: Knowledge-Based Skincare System")
st.write(f"Hello **{name}**. Running inference engine for **{skin_type}** skin targeting **{concern}**...")

if st.button("Generate Recommendations"):
    with st.spinner("Applying Knowledge Base rules... Parsing Ingredients..."):
        results = engine.inference(
            skin_type=skin_type,
            concern=concern,
            product_type=p_type,
            max_price=max_price
        )

    if not results.empty:
        st.success(f"Analysis Complete. Found {len(results)} matches based on ingredient logic.")
        
        for i, row in results.iterrows():
            with st.container():
                # Card Layout
                c1, c2 = st.columns([1, 4])
                
                with c1:
                    if pd.notna(row['picture_src']):
                        st.image(row['picture_src'], use_container_width=True)
                    else:
                        st.text("No Image")
                
                with c2:
                    # Header with Score
                    st.markdown(f"### {row['product_name']} <span class='score-badge'>Score: {row['expert_score']}</span>", unsafe_allow_html=True)
                    st.caption(f"**{row['brand']}** | {row['product_type']} | Rp {row['price_cleaned']:,.0f}")
                    
                    # XAI SECTION (The Explainable AI part)
                    st.info(f"**💡 Why this is recommended:**\n\n{row['explanation']}")
                    
                    with st.expander("View Product Details"):
                        st.write(row['description'])
                
                st.markdown("---")
    else:
        st.warning("No products met the strict criteria. Try increasing the price range or changing the product type.")

else:
    st.info("Awaiting input data to run the Knowledge Inference Engine.")