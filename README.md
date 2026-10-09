
# AI Customer Churn Intelligence Platform

An end-to-end **Data Science + Explainable AI + Generative AI** application that predicts telecom customer churn, explains predictions with SHAP, retrieves relevant retention policies using RAG, and generates AI-powered customer retention recommendations.

The project combines **Machine Learning, FastAPI, Streamlit, PostgreSQL, SHAP, RAG, and Google Gemini** into a complete deployed application.

## Live Demo

| Resource | Link |
|---|---|
| **Live Streamlit Application** | [Open Live Dashboard](https://customer-churn-ai-7rfwbadcfjrzupnq6zxh78.streamlit.app/) |
| **FastAPI Backend** | [Open Backend API](https://customer-churn-api-qjoz.onrender.com/) |
| **Swagger API Documentation** | [Try Prediction API](https://customer-churn-api-qjoz.onrender.com/docs) |
| **GitHub Repository** | [View Source Code](https://github.com/Akshit-Singh-00/customer-churn-ai) |

> **Note:** The backend is hosted on Render's free tier, so the first request may take additional time after inactivity. If Gemini is temporarily unavailable, the ML prediction, SHAP explanations, and retrieved retention policies remain accessible.

---

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Predicting churn helps businesses identify customers at risk of leaving and take appropriate retention actions.

This project develops an AI-powered churn intelligence system that:

1. Predicts whether a telecom customer is likely to churn.
2. Calculates the customer's churn probability.
3. Categorizes customers into different risk levels.
4. Explains model predictions using SHAP.
5. Retrieves relevant customer retention policies using RAG.
6. Generates personalized retention recommendations using Gemini.
7. Saves prediction history in a database.
8. Displays customer insights through an interactive dashboard.

The objective is to demonstrate how Data Science and Generative AI can work together to solve a real-world business problem.

---

## Key Features

### 1. Customer Churn Prediction

- Predicts whether a customer is likely to churn.
- Generates churn probability scores.
- Uses a trained machine learning classification model.
- Applies a tuned decision threshold.

### 2. Customer Risk Categorization

Customers are categorized into four levels:

- **Low Risk**
- **Medium Risk**
- **High Risk**
- **Critical Risk**

These categories help identify customers who may require retention attention.

### 3. Explainable AI with SHAP

- Explains individual customer predictions.
- Identifies features contributing to increased churn risk.
- Identifies features contributing to reduced churn risk.
- Helps interpret the model's decision-making process.

### 4. AI-Powered Retention Recommendations

- Integrates Google's Gemini API.
- Generates customer risk summaries.
- Explains important risk factors.
- Suggests practical retention actions.
- Uses fallback handling when Gemini is unavailable.

### 5. RAG-Based Retention Policy Retrieval

- Retrieves relevant company retention-policy passages.
- Uses TF-IDF for document vectorization.
- Uses cosine similarity for ranking relevant policies.
- Provides retrieved policies as context to Gemini.
- Displays relevant policy information in the dashboard.

### 6. Prediction History

- Stores previous predictions in a database.
- Records churn probabilities.
- Stores customer account information and risk categories.
- Tracks the status of AI-generated recommendations.

### 7. Analytics Dashboard

- Displays recent customer predictions.
- Identifies high-risk customers.
- Shows risk distribution.
- Shows prediction distribution.
- Calculates average churn probability.
- Visualizes average churn risk by contract type.

### 8. Cloud Deployment

- FastAPI backend deployed on Render.
- PostgreSQL database hosted on Render.
- Streamlit frontend deployed on Streamlit Community Cloud.

---

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Model Evaluation | Cross-Validation, Precision, Recall, F1, ROC-AUC |
| Explainable AI | SHAP |
| Generative AI | Google Gemini API |
| RAG | TF-IDF, Cosine Similarity |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Database | SQLite, PostgreSQL |
| ORM | SQLAlchemy |
| Model Serialization | Joblib |
| Version Control | Git, GitHub |
| Cloud Deployment | Render, Streamlit Community Cloud |

---

## 🏗️ System Architecture

```text
                    Customer
                       |
                       v
              Streamlit Dashboard
                       |
                       v
                 FastAPI Backend
                       |
                       v
                Input Validation
                       |
                       v
                Feature Engineering
                       |
                       v
              ML Prediction Pipeline
                       |
              +--------+--------+
              |                 |
              v                 v
       Churn Probability    SHAP Explanation
              |                 |
              +--------+--------+
                       |
                       v
                 Risk Category
                       |
                       v
              RAG Policy Retrieval
                       |
                       v
               Relevant Policies
                       |
                       v
               Gemini AI Assistant
                       |
                       v
            Retention Recommendations
                       |
                       v
                PostgreSQL Database
                       |
                       v
             Prediction History
                       |
                       v
              Analytics Dashboard
```

The application separates machine learning predictions, model explanations, policy retrieval, and AI-generated recommendations into different components.

This modular architecture makes the system easier to debug, maintain, and deploy.

---

## Dataset

The project uses the **IBM Telco Customer Churn Dataset**.

Dataset: [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information related to:

- Customer demographics
- Customer tenure
- Phone services
- Internet services
- Technical support
- Contract type
- Payment methods
- Monthly charges
- Total charges
- Customer churn status

### Target Variable

`Churn`

```text
Churn = Yes → Customer leaves

Churn = No  → Customer stays
```

The machine learning problem is formulated as **Binary Classification**.

---

## Machine Learning Workflow

### Step 1 — Data Understanding

Explored the dataset using Pandas.

Analyzed:

- Dataset dimensions
- Column names
- Data types
- Missing values
- Duplicate records
- Target distribution

### Step 2 — Data Cleaning

Performed the following operations:

- Standardized column names.
- Converted `TotalCharges` into a numeric column.
- Handled blank values using domain-aware logic.
- Checked duplicate records.
- Validated numerical columns.
- Converted the churn target into binary values.

### Step 3 — Exploratory Data Analysis

Investigated churn patterns across:

- Contract type
- Customer tenure
- Monthly charges
- Total charges
- Internet service
- Technical support
- Payment method
- Online security
- Customer demographics

This helped identify potential factors associated with customer churn.

### Step 4 — Feature Engineering

Created additional features such as:

```text
tenure_group
total_services
avg_monthly_spend
automatic_payment
is_month_to_month
has_tech_support
```

These features provide additional information about customer relationships, services, and payment behavior.

### Step 5 — Train/Test Split

Used stratified train/test splitting to preserve the churn class distribution.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Step 6 — Preprocessing Pipeline

Used:

- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`
- `Pipeline`

Preprocessing was fitted using training data to reduce the risk of data leakage.

### Step 7 — Model Training

Experimented with:

1. Logistic Regression
2. Balanced Logistic Regression
3. Random Forest
4. XGBoost

### Step 8 — Model Evaluation

Compared model performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Confusion Matrix

Used stratified cross-validation for model comparison.

### Step 9 — Hyperparameter Tuning

Used `RandomizedSearchCV` to search for improved model hyperparameters.

### Step 10 — Threshold Optimization

Evaluated different decision thresholds to understand precision-recall tradeoffs.

The deployed model uses a saved classification threshold.

### Step 11 — Model Serialization

Saved the complete trained pipeline using Joblib.

```text
models/
├── churn_model.joblib
└── churn_threshold.joblib
```

This allows FastAPI to load the trained model without retraining it for every request.

### Final Model Evaluation

The exact held-out test metrics should be taken from the model experimentation notebook.

| Metric | Result |
|---|---|
| Accuracy | Add measured value |
| Precision | Add measured value |
| Recall | Add measured value |
| F1 Score | Add measured value |
| ROC-AUC | Add measured value |
| Average Precision | Add measured value |
| Deployed Decision Threshold | 0.55 |

---

## Explainable AI with SHAP

SHAP is used to explain individual predictions.

Instead of only returning:

```text
Prediction: Churn

Churn Probability: 89.44%
```

the application also provides factors that influence the prediction.

Example:

```text
Churn Probability: 89.44%

Prediction: Churn

Risk Category: Critical

Factors Increasing Churn Risk:

- Low customer tenure
- Month-to-month contract
- Fiber optic internet service
- Short customer relationship
```

The explanations help users understand which features contributed to the model's output.

**Important:** SHAP explains model behavior and associations. It does not prove that a feature directly causes customer churn.

---

## Generative AI Retention Assistant

The project integrates Google's Gemini API to generate natural-language customer retention recommendations.

The LLM receives:

- Customer information
- Predicted churn probability
- Risk category
- Important SHAP factors
- Retrieved retention policies

It then produces:

1. Customer Risk Summary
2. Main Risk Factors
3. Recommended Retention Actions
4. Intervention Priority

The machine learning model remains responsible for calculating the prediction and probability.

Gemini is used for explanation and recommendations.

### AI Failure Handling

If Gemini becomes unavailable, the system returns:

```text
AI recommendation is temporarily unavailable.

The machine-learning prediction and SHAP
explanation are still available.
```

This ensures that the core churn prediction service continues functioning even when the external LLM is unavailable.

---

## Retrieval-Augmented Generation (RAG)

The project includes a lightweight RAG system for customer retention-policy retrieval.

### RAG Workflow

```text
Customer Information
        |
        v
Build Retrieval Query
        |
        v
TF-IDF Vectorization
        |
        v
Cosine Similarity
        |
        v
Retrieve Relevant Policies
        |
        v
Send Policies to Gemini
        |
        v
Generate Retention Advice
```

### Example Retention Policies

The demonstration policy document contains examples such as:

- Contract migration discounts
- Complimentary technical support
- Billing plan reviews
- New customer onboarding
- Automatic payment incentives
- Retention restrictions

The AI assistant uses retrieved policy passages to help generate recommendations consistent with the provided context.

### Policy Transparency

The Streamlit dashboard includes a section called:

**View Retention Policies Used**

This allows users to inspect the policies retrieved by the RAG system.

The retention policies are synthetic demonstration examples and do not represent real authorized company offers.

---

## FastAPI Backend

The backend is built using FastAPI.

### Live Backend

[https://customer-churn-api-qjoz.onrender.com](https://customer-churn-api-qjoz.onrender.com)

### Swagger Documentation

[https://customer-churn-api-qjoz.onrender.com/docs](https://customer-churn-api-qjoz.onrender.com/docs)

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Generate churn prediction and explanation |
| GET | `/history` | Retrieve recent predictions |
| GET | `/history/high-risk` | Retrieve High and Critical risk predictions |

---

## Example API Request

```json
{
  "gender": "Female",
  "seniorcitizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 5,
  "phoneservice": "Yes",
  "multiplelines": "No",
  "internetservice": "Fiber optic",
  "onlinesecurity": "No",
  "onlinebackup": "No",
  "deviceprotection": "No",
  "techsupport": "No",
  "streamingtv": "Yes",
  "streamingmovies": "Yes",
  "contract": "Month-to-month",
  "paperlessbilling": "Yes",
  "paymentmethod": "Electronic check",
  "monthlycharges": 89.5,
  "totalcharges": 447.5
}
```

### Example Response

A shortened response from the deployed API:

```json
{
  "churn_probability": 0.8944,
  "churn_probability_percent": 89.44,
  "prediction": "Churn",
  "risk_category": "Critical",
  "threshold": 0.55,
  "top_risk_factors": [
    {
      "feature": "tenure",
      "impact": 0.0541
    },
    {
      "feature": "is_month_to_month",
      "impact": 0.0447
    }
  ],
  "ai_status": "unavailable",
  "prediction_id": 1
}
```

The full API response also includes protective SHAP factors, retrieved policies, and the AI recommendation or fallback message.

---

## Database Integration

The application uses SQLAlchemy for database operations.

### Local Development

SQLite

### Cloud Deployment

PostgreSQL hosted on Render

The system stores information such as:

- Prediction ID
- Prediction timestamp
- Customer tenure
- Contract type
- Internet service
- Technical support
- Payment method
- Monthly charges
- Total charges
- Churn probability
- Prediction result
- Risk category
- SHAP factors
- AI recommendation status
- AI recommendation text

### Database Workflow

```text
Customer Prediction
        |
        v
FastAPI
        |
        v
Machine Learning Prediction
        |
        v
SQLAlchemy
        |
        v
PostgreSQL Database
        |
        v
Prediction History
```

This allows predictions to remain available across application sessions.

---

## Streamlit Analytics Dashboard

The Streamlit frontend provides an interactive interface for customer churn analysis.

### Features

- Customer information form
- Churn prediction button
- Churn probability display
- Customer risk categorization
- SHAP explanation tables
- AI retention recommendation
- Retrieved policy viewer
- Recent prediction history
- High-risk prediction history
- Interactive Plotly visualizations

### Analytics

The dashboard provides:

**Risk Distribution**

Shows how saved predictions are distributed across risk categories.

**Prediction Distribution**

Compares the number of Churn and Stay predictions.

**Average Churn Risk by Contract**

Displays average predicted churn probability across contract categories.

**Prediction History**

Displays recent customer prediction records stored in the database.

### Live Dashboard

[Launch AI Customer Churn Intelligence Platform](https://customer-churn-ai-7rfwbadcfjrzupnq6zxh78.streamlit.app/)

---

## Project Structure

```text
customer-churn-ai/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── database.py
│   │   ├── db_models.py
│   │   │
│   │   └── services/
│   │       ├── feature_engineering.py
│   │       ├── prediction.py
│   │       ├── explainability.py
│   │       ├── ai_retention.py
│   │       ├── rag_retention.py
│   │       └── history.py
│   │
│   └── dashboard/
│       ├── streamlit_app.py
│       ├── api_client.py
│       └── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── retention_policies/
│       └── retention_policies.txt
│
├── models/
│   ├── churn_model.joblib
│   └── churn_threshold.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_experiments.ipynb
│   └── 05_shap_explainability.ipynb
│
├── backend-requirements.txt
├── requirements.txt
├── render.yaml
├── .python-version
├── .gitignore
└── README.md
```

---

## Run the Project Locally

### Prerequisites

- Python compatible with the saved ML model
- Git
- pip
- Gemini API key (optional for AI recommendations)

### Step 1 — Clone Repository

```bash
git clone https://github.com/Akshit-Singh-00/customer-churn-ai.git

cd customer-churn-ai
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv
```

On Windows:

```powershell
venv\Scripts\activate
```

### Step 3 — Install Dependencies

Install backend dependencies:

```bash
pip install -r backend-requirements.txt
```

Install frontend dependencies:

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root.

```env
FASTAPI_URL=http://127.0.0.1:8000
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=your_available_gemini_model
```

`DATABASE_URL` can also be configured for PostgreSQL.

Without it, the application uses SQLite locally.

Never upload real API keys or database credentials to GitHub.

### Step 5 — Start FastAPI

```bash
python -m uvicorn app.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Step 6 — Start Streamlit

Open a new terminal and activate the same virtual environment.

Run:

```bash
python -m streamlit run app/dashboard/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

You can now enter customer information and generate predictions.

---

## Cloud Deployment

The application is deployed using two cloud platforms.

### Render

Hosts:

- FastAPI backend
- PostgreSQL database

Backend URL:

[https://customer-churn-api-qjoz.onrender.com](https://customer-churn-api-qjoz.onrender.com)

### Streamlit Community Cloud

Hosts:

- Customer prediction interface
- SHAP explanation display
- Retention recommendations
- Prediction history
- Plotly analytics

Frontend URL:

[https://customer-churn-ai-7rfwbadcfjrzupnq6zxh78.streamlit.app/](https://customer-churn-ai-7rfwbadcfjrzupnq6zxh78.streamlit.app/)

### Production Architecture

```text
Streamlit Community Cloud
           |
           v
     Render FastAPI
           |
    +------+------+
    |             |
    v             v
 ML + SHAP       RAG
    |             |
    +------+------+
           |
           v
         Gemini
           |
           v
    Render PostgreSQL
```

---

## What I Learned

Building this project helped me understand how Data Science, Machine Learning, Generative AI, and software development work together in a real application.

### 1. Data Cleaning and Exploratory Data Analysis

I learned how to:

- Inspect datasets using Pandas.
- Identify incorrect column data types.
- Handle missing or blank values.
- Validate numerical and categorical features.
- Identify important patterns using visualizations.
- Interpret relationships between customer features and churn.

### 2. Machine Learning Model Development

I learned how to:

- Perform feature engineering.
- Split datasets correctly.
- Prevent data leakage.
- Build preprocessing pipelines.
- Train multiple classification models.
- Handle class imbalance.
- Evaluate models using multiple metrics.
- Apply cross-validation.
- Tune hyperparameters.
- Optimize decision thresholds.
- Save trained models for deployment.

### 3. Explainable Artificial Intelligence

I learned how SHAP helps explain machine learning predictions.

Instead of only obtaining a prediction, I learned how to analyze which features contribute to increasing or decreasing the model's predicted churn risk.

I also learned that SHAP explanations describe model behavior rather than proving causal relationships.

### 4. Generative AI Integration

I learned how to connect a machine learning application with Google's Gemini API.

This included:

- Building structured prompts.
- Passing customer information to an LLM.
- Generating readable customer summaries.
- Requesting retention recommendations.
- Handling unavailable API services.

### 5. Retrieval-Augmented Generation

I learned the fundamentals of RAG by implementing:

- Policy document loading
- TF-IDF vectorization
- Cosine similarity
- Relevant document retrieval
- Retrieval-grounded AI prompts

This helped me understand how external knowledge can be incorporated into LLM-generated responses.

### 6. Backend Development

I learned how to build REST APIs using FastAPI.

This included:

- Creating API endpoints.
- Validating inputs using Pydantic.
- Loading trained ML models.
- Integrating SHAP explanations.
- Connecting RAG and Gemini.
- Returning structured JSON responses.
- Using Swagger for API testing.

### 7. Database Integration

I learned how to:

- Use SQLite during local development.
- Use SQLAlchemy ORM.
- Create database tables.
- Store model predictions.
- Retrieve historical records.
- Connect FastAPI to PostgreSQL.
- Use environment variables for database configuration.

### 8. Dashboard Development

I learned how to create an interactive Streamlit dashboard.

This included:

- Customer input forms
- API integration
- Prediction results
- Risk indicators
- SHAP explanation tables
- AI recommendation displays
- Plotly charts
- Prediction history
- Analytics dashboards

### 9. Cloud Deployment

I learned how to:

- Deploy FastAPI to Render.
- Deploy PostgreSQL to Render.
- Deploy Streamlit using Streamlit Community Cloud.
- Configure environment variables.
- Handle cloud dependency issues.
- Debug deployment logs.
- Connect separately hosted frontend and backend services.
- Manage application changes through GitHub.

---

## Challenges Faced and How I Solved Them

During development, I faced several technical challenges that helped improve my debugging and problem-solving skills.

### 1. KeyError During Data Cleaning

**Problem:**

While accessing the `totalcharges` column, I received:

```text
KeyError: 'totalcharges'
```

**Cause:**

The dataset contained column names with different capitalization.

**Solution:**

I standardized the column names:

```python
df_clean.columns = (
    df_clean.columns
    .str.strip()
    .str.lower()
)
```

**What I Learned:**

Consistent naming conventions and correct notebook execution order are important for reliable data processing.

### 2. NameError in Jupyter Notebooks

**Problem:**

I encountered errors such as:

```text
NameError: name 'y_train' is not defined

NameError: name 'preprocessor' is not defined

NameError: name 'shap_churn' is not defined
```

**Cause:**

Variables were not initialized in the current notebook session, or the Jupyter kernel had restarted.

**Solution:**

I recreated required variables and executed notebook cells in the correct order.

**What I Learned:**

Jupyter notebook variables exist in temporary memory. Separate notebooks do not automatically share their variables.

### 3. Gemini API Server Errors

**Problem:**

Gemini returned:

```text
503 UNAVAILABLE

This model is currently experiencing high demand.
```

**Cause:**

Temporary model unavailability or high demand.

**Solution:**

I implemented exception handling and fallback responses.

```python
try:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    ai_recommendation = response.text

except Exception:
    ai_recommendation = (
        "AI recommendation is temporarily unavailable. "
        "The ML churn prediction is still available."
    )
```

**What I Learned:**

External AI APIs can fail independently of the main application. A reliable system should provide fallback behavior rather than crashing.

### 4. Streamlit Variable Scope Errors

**Problem:**

I encountered:

```text
NameError: name 'data' is not defined
```

**Cause:**

Some dashboard components tried to access API response data before the user submitted the form.

**Solution:**

I moved response-dependent code inside the `if submitted:` block.

**What I Learned:**

Python indentation, variable scope, and Streamlit's script rerun behavior are important when building interactive applications.

### 5. FastAPI Internal Server Error

**Problem:**

After adding the database layer, the prediction endpoint returned:

```text
500 Internal Server Error
```

**Solution:**

I checked the database models, SQLAlchemy sessions, table creation, and prediction-saving logic.

After fixing the database setup and retesting, the API successfully returned a prediction ID.

**What I Learned:**

API testing should cover both the machine learning logic and the database operations.

### 6. Render Python Dependency Conflicts

**Problem:**

During deployment, Render failed while installing dependencies.

One error was:

```text
contourpy==1.4.0 requires Python >=3.12
```

while Render was configured for Python 3.11.

**Solution:**

I separated the backend dependencies from the complete local development environment.

I created:

```text
backend-requirements.txt
```

for FastAPI deployment.

**What I Learned:**

Production applications should install only the dependencies they actually require. Smaller dependency files make deployment easier and help avoid version conflicts.

### 7. Missing PostgreSQL Driver

**Problem:**

Render returned:

```text
ModuleNotFoundError: No module named 'psycopg'
```

**Cause:**

The PostgreSQL connection string expected the `psycopg` driver, but the required driver was not installed.

**Solution:**

I added:

```text
psycopg[binary]
```

to the backend dependencies.

**What I Learned:**

Database drivers must match the connection configuration used by SQLAlchemy.

### 8. Missing Machine Learning Model During Deployment

**Problem:**

Render returned:

```text
FileNotFoundError:

models/churn_model.joblib
```

**Cause:**

The trained model file was available locally but had not been included in the GitHub repository.

**Solution:**

I updated `.gitignore`, added the required model files to Git, and redeployed the backend.

**What I Learned:**

Files that exist locally are not automatically available in cloud deployment environments.

### 9. Missing SHAP Background Dataset

**Problem:**

Render could not locate:

```text
data/processed/telco_customer_churn_clean.csv
```

**Cause:**

The processed dataset required for SHAP background sampling was missing from the deployed repository.

**Solution:**

I added the processed dataset to GitHub and redeployed the application.

**What I Learned:**

Model explainability components may require additional runtime artifacts beyond the trained model itself.

### 10. Streamlit Cloud Installation Failure

**Problem:**

Streamlit Cloud failed while installing packages from the original large `requirements.txt`.

**Cause:**

The dependency file contained unnecessary notebook and machine learning packages, including packages incompatible with the deployment environment.

**Solution:**

I simplified the frontend requirements to:

```text
streamlit
pandas
plotly
requests
python-dotenv
```

and kept backend dependencies in a separate file.

**What I Learned:**

Frontend and backend applications should have separate deployment dependencies.

### 11. Undefined API Endpoint

**Problem:**

After Streamlit Cloud installed the dependencies, the application crashed with:

```text
NameError: name 'endpoint' is not defined
```

**Cause:**

The `predict_churn()` function attempted to send a request before defining the API endpoint URL.

**Solution:**

I constructed the URL inside the function:

```python
endpoint = f"{API_URL}/predict"
```

**What I Learned:**

Every function should initialize the values it needs, and deployment testing should include runtime behavior rather than only dependency installation.

---

## Limitations and Responsible Use

- The model is trained on a IBM sample telecom churn dataset.
- Churn probabilities are estimates, not guaranteed outcomes.
- SHAP explains model predictions, not causal relationships.
- TF-IDF retrieval may return partially relevant policy passages.
- Retention recommendations should be reviewed before real-world use.
- Demonstration retention policies are synthetic, not actual approved offers.
- Public prediction history endpoints are intended for demonstration and should not contain real customer information without authentication and appropriate privacy controls.
- Gemini recommendations depend on API availability.

---

## Project Highlights

This project demonstrates practical experience with:

- End-to-end Data Science
- Exploratory Data Analysis
- Machine Learning Classification
- Feature Engineering
- Model Evaluation
- Hyperparameter Tuning
- Explainable Artificial Intelligence
- Retrieval-Augmented Generation
- LLM Integration
- REST API Development
- PostgreSQL Integration
- Interactive Data Visualization
- Cloud Deployment
- Debugging and Problem Solving

The project helped me understand not only how to train a machine learning model, but also how to integrate it into an application that users can interact with through a deployed web interface.

---

## Author

**Akshit Singh**

GitHub: [Akshit-Singh-00](https://github.com/Akshit-Singh-00)

### Project Links

- [Live Streamlit Application](https://customer-churn-ai-7rfwbadcfjrzupnq6zxh78.streamlit.app/)
- [FastAPI Backend](https://customer-churn-api-qjoz.onrender.com/)
- [API Documentation](https://customer-churn-api-qjoz.onrender.com/docs)
- [Source Code](https://github.com/Akshit-Singh-00/customer-churn-ai)
