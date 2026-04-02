# E-Policing: Integrated Spatio-Temporal Crime Forecasting and Decision Support System

## Overview
E-Policing is a comprehensive, data-driven web application designed to assist law enforcement agencies in proactive crime prevention and resource management. Built with Flask and powered by machine learning, the system forecasts crime hotspots, analyzes spatial and temporal patterns, evaluates underlying risk factors, and optimizes the allocation of police officers across different regions (GN Divisions).

## Key Features
- **Crime Hotspot Prediction**: Predicts and visualizes high-risk areas for varying crime types using advanced machine learning models.
- **Spatio-Temporal Pattern Analysis**: Detects trends and patterns to identify when and where specific crimes are most likely to occur.
- **Risk Assessment Module**: Evaluates socio-economic and environmental factors contributing to crime in specific areas to guide strategic interventions.
- **Optimized Resource Allocation**: Automatically optimizes the deployment of police personnel by analyzing predicted demand and calculating the diminishing marginal returns of adding officers to different areas.
- **Past Crime Records**: View and filter historical crime data by type, month, and year for in-depth comparative analysis.
- **Interactive Dashboards**: Provides an intuitive, rich graphical interface to help decision-makers visualize resource distribution, risk areas on maps (GeoJSON), and overall projected crime reduction.

## Tech Stack
- **Backend:** Python, Flask, Flask-SQLAlchemy, Flask-Login
- **Machine Learning & Data Processing:** Pandas, XGBoost, PyTorch, TensorFlow, Scikit-learn, Joblib
- **Frontend:** HTML5, CSS3, JavaScript, Jinja2 Templates
- **Data Formats:** Parquet (Inference Data), GeoJSON (Map Data)

## Project Structure
```text
E-Policing/
├── app.py                  # Main Flask application entry point
├── extensions.py           # Shared Flask extensions (e.g., db)
├── requirements.txt        # Python dependency list
├── .env                    # Environment variables (not tracked by git)
├── routes/                 # Blueprint routes for auth, allocation, hotspot, pattern, risk, records
├── services/               # Core business and ML logic pipeline scripts
├── models/                 # Serialized ML models (*.pkl, *.joblib) and inference data
├── static/                 # Static assets (CSS, JS, images, map markers)
├── templates/              # Jinja2 HTML templates
└── utils/                  # Helper functions and utilities
```

## Setup & Installation

### Prerequisites
- Python 3.8 to 3.11 (Recommended)
- Relational Database (e.g., PostgreSQL,supabase)

### 1. Clone the repository
```bash
git clone <repository_url>
cd E-Policing
```

### 2. Set up a Virtual Environment
**Windows:**
```bash
python -m venv env
env\Scripts\activate
```
**Linux / macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project's root directory and add your environment specific details:
```env
SECRET_KEY=your_secure_secret_key
DB_URI=postgresql://user:password@localhost:5432/epolicing_db
```

### 5. Start the Application
```bash
python app.py
```
By default, the application will run in debug mode at `http://localhost:5000/`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
