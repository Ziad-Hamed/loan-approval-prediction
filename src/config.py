from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "Loan-Approval-Prediction.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.csv"
MODEL_PATH = MODELS_DIR / "best_svc_pipeline.joblib"

# Feature definitions
NUMERICAL_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "Total_Income",
    "LoanAmount",
    "Loan_Amount_Term",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]



# Build the preprocessing transformer
def build_preprocessor():
    """
    Build numerical and categorical preprocessing pipelines.

    Numerical:
        Median imputation -> Standard scaling

    Categorical:
        Most-frequent imputation -> One-hot encoding
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor