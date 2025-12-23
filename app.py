import streamlit as st
import pandas as pd
import plotly.express as px
# NOTICE THE CHANGE HERE: Hum class import kar rahe hain, function nahi
from src.data_engine import DataGenerator
from src.brain import CausalBrain
from src.marketer import EmailAgent

# --- PAGE CONFIG ---
st.set_page_config(page_title="ConversionAI", layout="wide", page_icon="🚀")

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS ---
st.sidebar.title("⚙️ Control Panel")
n_users = st.sidebar.slider("Sample Size", 1000, 10000, 5000)
groq_key = st.sidebar.text_input("Groq API Key (Optional)", type="password")

if st.sidebar.button("🚀 Run Simulation"):
    with st.spinner("Simulating Experiment & Training Models..."):
        # 1. Get Data (Using the NEW Class logic)
        gen = DataGenerator(n_samples=n_users)
        df = gen.get_data()
        
        # 2. Train AI
        brain = CausalBrain()
        features = ['Recency', 'Frequency', 'Amount', 'Age', 'Income']
        brain.train(df, features)
        
        # 3. Predict Lift
        df['Uplift_Score'] = brain.get_uplift(df, features)
        
        # Save to session
        st.session_state['data'] = df
        st.session_state['brain'] = brain
        st.success("Analysis Complete!")

# --- MAIN APP ---
st.title("🚀 ConversionAI: Causal Marketing Engine")
st.markdown("Use **Machine Learning** to find 'Persuadable' customers and **GenAI** to convert them.")

if 'data' in st.session_state:
    df = st.session_state['data']
    
    # TABS FOR CLEAN UI
    tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "🎯 Target Segmentation", "📧 AI Marketing Agent"])
    
    # --- TAB 1: ANALYSIS ---
    with tab1:
        st.subheader("Did the Coupon Work?")
        col1, col2 = st.columns(2)
        
        # Simple Logic: Avg Conversion with vs without coupon
        conv_with = df[df['Treatment']==1]['Conversion'].mean()
        conv_without = df[df['Treatment']==0]['Conversion'].mean()
        
        with col1:
            st.metric("Conversion (With Coupon)", f"{conv_with:.1%}")
        with col2:
            st.metric("Conversion (No Coupon)", f"{conv_without:.1%}")
            
        st.info(f"Observation: The coupon generally increases sales by {conv_with - conv_without:.1%}. But is it profitable for everyone?")

    # --- TAB 2: SEGMENTATION ---
    with tab2:
        st.subheader("Who should we target?")
        
        # Histogram of Uplift
        fig = px.histogram(df, x="Uplift_Score", color="Segment", nbins=50, 
                           title="Customer Segmentation based on Uplift Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Slider
        threshold = st.slider("Uplift Threshold (Target users above this score)", -0.5, 0.5, 0.1)
        
        # Apply Logic
        targeted_df = df[df['Uplift_Score'] > threshold]
        ignored_df = df[df['Uplift_Score'] <= threshold]
        
        st.write(f"🎯 Targeting **{len(targeted_df)}** customers who react positively.")
        st.dataframe(targeted_df[['Age', 'Income', 'Recency', 'Segment', 'Uplift_Score']])

    # --- TAB 3: GEN AI AGENT ---
    with tab3:
        st.subheader("📧 Auto-Generate Personalized Emails")
        
        if len(targeted_df) > 0:
            # Pick the best candidate (Highest Uplift)
            best_lead = targeted_df.sort_values(by='Uplift_Score', ascending=False).iloc[0]
            
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                st.markdown("### Best Lead Profile")
                st.write(f"**Segment:** {best_lead['Segment']}")
                st.write(f"**Age:** {best_lead['Age']}")
                st.write(f"**Last Visit:** {best_lead['Recency']} days ago")
                st.write(f"**Est. Spend:** ${int(best_lead['Amount'])}")
            
            with col_b:
                if st.button("✨ Write Email for this User"):
                    agent = EmailAgent(api_key=groq_key)
                    with st.spinner("AI is writing copy..."):
                        email_content = agent.write_email(best_lead)
                        st.text_area("Generated Draft:", email_content, height=250)
        else:
            st.warning("No targets selected! Lower the threshold in Tab 2.")
            
else:
    st.info("👈 Click 'Run Simulation' in the sidebar to start.")