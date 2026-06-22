# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
# pyrefly: ignore [missing-import]
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="IT Salary Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    h1, h2, h3, h4 {
        color: #a5b4fc;
    }
    .project-insight-card {
        background-color: rgba(255, 255, 255, 0.02);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4f46e5;
        margin-bottom: 20px;
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
        rule = alt.Chart(pd.DataFrame({'predicted': [predicted_salary]})).mark_rule(color='#fbbf24', size=4).encode(
            x='predicted:Q',
            tooltip=[alt.Tooltip('predicted:Q', title='Your Prediction', format=',.0f')]
        )
        return (base_chart + rule).properties(height=300)
    return base_chart.properties(height=300)

def main():
    data_objects, model, status_msg = load_models_and_data()
    if data_objects is None or model is None:
        st.error(status_msg)
        return
        
    df_orig = data_objects['df_original']

    st.title("AI Compensation Predictor")
    
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1, 1, 1])
    with btn_col1:
        st.write("Enter professional parameters below to estimate the annual compensation.")
    with btn_col2:
        if st.button("📊 Open Presentation", use_container_width=True):
            os.system('open "IT Specialists Salary Prediction_ An End-to-End Machine Learning Pipeline.pptx"')
    with btn_col3:
        st.link_button("📓 Kaggle Notebook", "https://www.kaggle.com/code/mokhlestarmiz00/exam-notebook", use_container_width=True)
    with btn_col4:
        st.link_button("🐙 GitHub Repo", "https://github.com/MokhlesTr/IT-salary-estimation-ML-project", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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

    st.markdown("---")
    
    # -------------------------------------------------------------
    # PROJECT DOCUMENTATION & PRESENTATION SUMMARY (MOVED TO MAIN PAGE)
    # -------------------------------------------------------------
    st.title("Project Summary & Methodology")
    st.write("A deep dive into the data engineering and machine learning techniques used to build this AI.")

    doc_col1, doc_col2 = st.columns(2)

    with doc_col1:
        st.markdown("""
        <div class="project-insight-card">
            <h4>1. Data Preprocessing & Cleaning</h4>
            <p><strong>Missing Values:</strong> Handled gracefully by imputing 'Unknown' for categorical variables to retain maximum dataset density.</p>
            <p><strong>Outliers Treatment:</strong> Applied the robust Interquartile Range (IQR) method to cap extreme salary anomalies.</p>
            <p><strong>Effect on Results:</strong> Capping extreme salaries drastically reduced the skewness of our target variable, allowing the model to learn general market trends rather than memorizing a few ultra-high-paying outliers.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="project-insight-card">
            <h4>3. Algorithm Selection & Optimization</h4>
            <p>We rigorously tested three algorithms:</p>
            <ul>
                <li><strong>Linear Regression:</strong> Used as our baseline model.</li>
                <li><strong>Random Forest:</strong> To capture non-linear relationships.</li>
                <li><strong>XGBoost (The Winner):</strong> An advanced distributed gradient boosting library.</li>
            </ul>
            <p><strong>Optimization:</strong> We utilized <code>GridSearchCV</code> with 5-fold cross-validation to find the absolute best hyperparameter combinations (depth, learning rate) for our models.</p>
        </div>
        """, unsafe_allow_html=True)

    with doc_col2:
        st.markdown("""
        <div class="project-insight-card">
            <h4>2. Feature Engineering (The Best Thing)</h4>
            <p>Instead of relying purely on raw data, we engineered synthetic features to help the AI find deeper mathematical correlations:</p>
            <ul>
                <li><strong>Interaction Variables:</strong> Created <code>Experience × Age</code> and <code>Projects × Experience</code>.</li>
                <li><strong>Categorization:</strong> Discretized continuous ages into logical bins.</li>
                <li><strong>Productivity:</strong> Derived a <code>Projects_per_year</code> ratio.</li>
            </ul>
            <p><strong>Effect on Results:</strong> Our engineered <code>Experience_Age</code> feature emerged as one of the most powerful predictive variables in the entire dataset!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="project-insight-card">
            <h4>4. Final Results & Insights</h4>
            <p><strong>Best Algorithm:</strong> XGBoost achieved the highest R² Score and the lowest Root Mean Squared Error (RMSE).</p>
            <p><strong>Surprising Finding:</strong> 'Job Satisfaction' and 'Experience_Age' consistently outranked basic variables like 'Education Level' in determining compensation.</p>
            <p><strong>Model Honesty:</strong> Due to utilizing complex algorithms on a smaller dataset (504 rows), we noted mild overfitting in training. This is a highly authentic constraint of real-world machine learning on limited data.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model Feature Importance")
    st.write("Visualizing the exact parameters that drive the XGBoost algorithm's predictions.")
    st.altair_chart(plot_feature_importance(model, data_objects['feature_names']), use_container_width=True)

if __name__ == "__main__":
    main()
