import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="IT Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indigo Theme
st.markdown("""
<style>
    /* Primary Indigo Accents */
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        color: white;
        border: none;
    }
    .prediction-container {
        padding: 2.5rem;
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(79, 70, 229, 0.1), rgba(67, 56, 202, 0.05));
        border: 1px solid rgba(79, 70, 229, 0.3);
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        color: #818cf8;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(129, 140, 248, 0.3);
    }
    .prediction-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #a5b4fc;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    try:
        with open('prepared_data.pkl', 'rb') as f:
            data = pickle.load(f)
            
        if os.path.exists('xgboost_model.pkl'):
            with open('xgboost_model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_name = "XGBoost"
        elif os.path.exists('random_forest_model.pkl'):
            with open('random_forest_model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_name = "Random Forest"
        else:
            return None, None, "No trained models found."
            
        return data, model, f"Active Model: {model_name}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def process_input(input_dict, data_objects):
    le_dict = data_objects['label_encoders']
    scaler = data_objects['scaler']
    
    exp = input_dict['Experience (years)']
    exp_cat = 'Junior' if exp <= 2 else 'Mid' if exp <= 5 else 'Senior' if exp <= 10 else 'Expert' if exp <= 20 else 'Veteran'
        
    age = input_dict['Age']
    age_cat = '20-25' if age <= 25 else '26-35' if age <= 35 else '36-45' if age <= 45 else '46-55' if age <= 55 else '56+'
        
    proj = input_dict['Number of projects']
    projects_per_year = proj / (exp + 1)
    experience_age = exp * age
    projects_experience = proj * exp
    
    encoded_features = {}
    cat_cols = ['Position', 'Specialization', 'Continent', 'Gender', 'Education', 'Type of employment']
    
    for col in cat_cols:
        try:
            encoded_features[f'{col}_encoded'] = le_dict[col].transform([input_dict[col]])[0]
        except ValueError:
            encoded_features[f'{col}_encoded'] = 0 
            
    try:
        exp_mapping = {'Junior': 1, 'Mid': 3, 'Senior': 4, 'Expert': 0, 'Veteran': 5}
        age_mapping = {'20-25': 0, '26-35': 1, '36-45': 2, '46-55': 3, '56+': 4}
        exp_encoded = exp_mapping.get(exp_cat, 0)
        age_encoded = age_mapping.get(age_cat, 0)
    except Exception:
        exp_encoded = 0
        age_encoded = 0

    features = {
        'Experience (years)': exp, 'Age': age, 'Number of projects': proj, 'Job satisfaction': input_dict['Job satisfaction'],
        'Position_encoded': encoded_features['Position_encoded'], 'Specialization_encoded': encoded_features['Specialization_encoded'],
        'Continent_encoded': encoded_features['Continent_encoded'], 'Gender_encoded': encoded_features['Gender_encoded'],
        'Education_encoded': encoded_features['Education_encoded'], 'Type of employment_encoded': encoded_features['Type of employment_encoded'],
        'Experience_category_encoded': exp_encoded, 'Projects_per_year': projects_per_year, 'Age_group_encoded': age_encoded,
        'Experience_Age': experience_age, 'Projects_Experience': projects_experience
    }
    
    feature_names = data_objects['feature_names']
    X = pd.DataFrame([features])[feature_names]
    numeric_features = data_objects['numeric_features']
    X[numeric_features] = scaler.transform(X[numeric_features])
    return X

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=False).head(10)
    df_imp['Feature'] = df_imp['Feature'].str.replace('_encoded', '').str.replace('_', ' ')
    
    chart = alt.Chart(df_imp).mark_bar(color='#6366f1', cornerRadiusEnd=4).encode(
        x=alt.X('Importance:Q', title='Importance Score'),
        y=alt.Y('Feature:N', sort='-x', title='Feature'),
        tooltip=['Feature', alt.Tooltip('Importance', format='.4f')]
    ).properties(height=350)
    return chart

def plot_salary_distribution(df, predicted_salary=None):
    base_chart = alt.Chart(df).mark_bar(color='#4f46e5', opacity=0.7).encode(
        x=alt.X('Salary ($/year):Q', bin=alt.Bin(maxbins=30), title='Salary ($)'),
        y=alt.Y('count():Q', title='Number of Professionals'),
        tooltip=['count()']
    )
    
    if predicted_salary is not None:
        rule = alt.Chart(pd.DataFrame({'predicted': [predicted_salary]})).mark_rule(color='#fbbf24', size=3).encode(
            x='predicted:Q',
            tooltip=[alt.Tooltip('predicted:Q', title='Your Prediction', format=',.0f')]
        )
        return (base_chart + rule).properties(height=300)
    return base_chart.properties(height=300)

def plot_salary_by_continent(df):
    chart = alt.Chart(df).mark_boxplot(color='#818cf8', size=40).encode(
        x=alt.X('Continent:N', title=''),
        y=alt.Y('Salary ($/year):Q', title='Salary ($)'),
        color=alt.Color('Continent:N', legend=None, scale=alt.Scale(scheme='indigo'))
    ).properties(height=350)
    return chart

def plot_salary_vs_experience(df):
    chart = alt.Chart(df).mark_circle(size=60, opacity=0.6, color='#6366f1').encode(
        x=alt.X('Experience (years):Q', title='Years of Experience'),
        y=alt.Y('Salary ($/year):Q', title='Salary ($)'),
        tooltip=['Position', 'Experience (years)', 'Salary ($/year)']
    ).properties(height=350)
    return chart

def main():
    data_objects, model, status_msg = load_models_and_data()
    if data_objects is None or model is None:
        st.error(status_msg)
        return
        
    df_orig = data_objects['df_original']

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #a5b4fc;'>IT Salary AI</h2>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio("Navigation", ["🎯 AI Predictor", "📊 Market Dashboard", "🧠 Model Analysis", "📚 Methodology"])
        st.markdown("---")
        st.caption(f"🔧 {status_msg}")
        st.caption(f"📈 Dataset: {len(df_orig)} records")

    if menu == "🎯 AI Predictor":
        st.title("AI Compensation Predictor")
        st.write("Enter professional parameters below to estimate the annual compensation.")
        
        col1, col2 = st.columns(2)
        with col1:
            position = st.selectbox("Job Position", options=sorted(df_orig['Position'].unique()))
            specialization = st.selectbox("Specialization Area", options=sorted(df_orig['Specialization'].unique()))
            experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
            num_projects = st.number_input("Number of Projects Completed", min_value=0, max_value=500, value=10)
            job_satisfaction = st.slider("Job Satisfaction Rating (0-100)", min_value=0, max_value=100, value=75)

        with col2:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            continent = st.selectbox("Continent", options=sorted(df_orig['Continent'].unique()))
            gender = st.selectbox("Gender", options=sorted(df_orig['Gender'].unique()))
            education = st.selectbox("Highest Education Level", options=sorted(df_orig['Education'].unique()))
            employment_type = st.selectbox("Employment Type", options=sorted(df_orig['Type of employment'].unique()))

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("CALCULATE PREDICTED SALARY", type="primary", use_container_width=True):
            user_input = {
                'Experience (years)': experience, 'Age': age, 'Number of projects': num_projects,
                'Job satisfaction': job_satisfaction, 'Position': position, 'Specialization': specialization,
                'Continent': continent, 'Gender': gender, 'Education': education, 'Type of employment': employment_type
            }
            
            with st.spinner("Processing input and executing model..."):
                X_processed = process_input(user_input, data_objects)
                prediction = model.predict(X_processed)[0]
                
                # Show Result and Chart side by side
                res_col, chart_col = st.columns([1, 1.5])
                with res_col:
                    st.markdown(f"""
                    <div class="prediction-container">
                        <div class="prediction-subtitle">Estimated Annual Compensation</div>
                        <div class="prediction-value">${prediction:,.2f}</div>
                        <div class="prediction-subtitle">Predicted using {status_msg.split()[-1]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with chart_col:
                    st.markdown("**Market Position**")
                    st.caption("The yellow line represents your predicted salary compared to the entire market dataset.")
                    st.altair_chart(plot_salary_distribution(df_orig, predicted_salary=prediction), use_container_width=True)

    elif menu == "📊 Market Dashboard":
        st.title("Global IT Market Dashboard")
        st.write("Explore the underlying trends in the IT professional dataset.")
        
        dash_col1, dash_col2 = st.columns(2)
        with dash_col1:
            st.subheader("Salary Distribution by Continent")
            st.altair_chart(plot_salary_by_continent(df_orig), use_container_width=True)
        with dash_col2:
            st.subheader("Salary vs. Experience")
            st.altair_chart(plot_salary_vs_experience(df_orig), use_container_width=True)

        st.subheader("Overall Salary Distribution")
        st.altair_chart(plot_salary_distribution(df_orig), use_container_width=True)

    elif menu == "🧠 Model Analysis":
        st.title("Model Feature Importance")
        st.write("Understand which parameters carry the most weight in the AI's decision-making process.")
        
        feat_col, text_col = st.columns([2, 1])
        with feat_col:
            st.altair_chart(plot_feature_importance(model, data_objects['feature_names']), use_container_width=True)
        with text_col:
            st.info("""
            **Analysis Insights:**
            - Features at the top strictly dictate the predicted salary.
            - **Experience_Age** (an engineered interaction feature) heavily outperforms individual base features.
            - **Job Satisfaction** plays a surprising but significant role in compensation variance.
            """)

    elif menu == "📚 Methodology":
        st.title("Project Methodology & Documentation")
        
        st.subheader("1. Data Preprocessing & Cleaning")
        st.write("""
        - **Missing Values:** Imputed 'Unknown' for categorical variables to prevent data loss.
        - **Outliers:** Applied Interquartile Range (IQR) method to cap extreme salary anomalies.
        - **Impact:** Reduced skewness and improved generalized learning.
        """)

        st.subheader("2. Feature Engineering")
        st.write("""
        - **Interaction Terms:** Created complex variables like `Experience × Age`.
        - **Categorization:** Discretized continuous age and experience into logical bins.
        - **Productivity:** Created `Projects_per_year` ratio.
        """)

        st.subheader("3. Algorithm Selection")
        st.write("""
        Evaluated Linear Regression, Random Forest, and XGBoost.
        - Used **GridSearchCV** with 5-fold Cross-Validation for hyperparameter tuning.
        - **XGBoost** was selected as the optimal model for achieving the highest R² Score.
        """)

if __name__ == "__main__":
    main()
