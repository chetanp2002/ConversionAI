# 🚀 ConversionAI: Causal Marketing Engine  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)  
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)  
![GenAI](https://img.shields.io/badge/GenAI-Llama3-purple)  

### 🔍 Overview  
**ConversionAI** is an advanced **Causal Inference & GenAI** platform designed to optimize marketing **ROI**. Unlike traditional churn models that simply predict *who will buy*, this engine uses **Uplift Modeling** to identify customers who will be *persuaded* to buy only if given an incentive (Persuadables).  

It effectively filters out:  
- **Sure Things:** Customers who would buy anyway (saving wasted budget).  
- **Sleeping Dogs:** Customers who react negatively to marketing (preventing churn).  

Once targets are identified, the system employs a **Generative AI Agent (Llama-3)** to draft hyper-personalized outreach emails automatically.  

### 🛠️ Tech Stack  
- **Core Logic:** Causal Inference (Uplift Modeling / **ITE** Estimation)  
- **Machine Learning:** XGBoost (T-Learner Architecture)  
- **Generative AI:** Groq **API** (Llama-3 8B) for automated copywriting  
- **Visualization:** Plotly Express & Streamlit  
- **Data Engineering:** Pandas & NumPy (Synthetic **RFM** Data Generation)  

### 🧠 The Science: T-Learner Architecture  
The engine implements a **Two-Model (T-Learner)** approach to estimate the **Individual Treatment Effect (**ITE**)**:  

$$\text{Uplift} = P(\text{Buy} | \text{Treatment}) - P(\text{Buy} | \text{Control})$$  

1. **Control Model:** Trained on customers who received NO coupon.  
2. **Treatment Model:** Trained on customers who received A coupon.  
3. **Lift Calculation:** The difference between the two predictions determines the user's segment.  

### 📚 Acknowledgements & Inspiration  
While this project implements a custom **T-Learner using XGBoost** for lightweight deployment, the causal logic draws inspiration from:  
- **Microsoft DoWhy:** For the structural causal model framework.  
- **Microsoft EconML:** For advanced heterogeneous treatment effect estimation.  
*Note: This project uses a manual implementation to demonstrate the underlying mathematics without heavy dependencies.*  

### 📂 Project Structure  
```text  
ConversionAI/  
├── app.py                 # Main Streamlit Dashboard Application  
├── requirements.txt       # Project Dependencies  
└── src/  
    ├── __init__.py  
    ├── data_engine.py     # Generates realistic **RFM** data with hidden causal patterns  
    ├── brain.py           # The ML Engine (XGBoost T-Learner implementation)  
    └── marketer.py        # GenAI Agent (Connects to Groq/Llama-3)  
```
### How to Run Locally  
Clone the Repository  

```bash 

git clone [https://github.com/yourusername/ConversionAI.git](https://github.com/yourusername/ConversionAI.git) 
``` 
Install Dependencies  

```Bash  

pip install -r requirements.txt  
```
Run the Application  

```Bash  

streamlit run app.py  
```

Access the Dashboard Open your browser to [http://localhost:**8501**.](http://localhost:**8501**.) (Optional: Enter a free Groq **API** Key in the sidebar to enable the GenAI Email Agent)  

### 📊 Key Features  
Real-time **ROI** Simulation: Adjust uplift thresholds to see impact on net profit.  

Customer Segmentation: Visualizes the distribution of *Persuadables* vs *Sleeping Dogs*.  

Automated Action: One-click generation of personalized marketing copy for top leads.  

### 📜 License  
This project is open-source and available under the **MIT** License.  