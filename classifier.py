import pandas as pd
import plotly.express as pe
import plotly.graph_objects as go
import os
import dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

dotenv.load_dotenv()


class TravelTechAnalysis:
    def __init__(self):
        self.engine = self.connect_to_db()
        self.df = self.fetch_data()
        self.models = {}

    def connect_to_db(self):
        """Connects to MySQL database using credentials from environment variables."""
        try:
            password = os.getenv("password_db").replace("@", "%40")
            user = os.getenv("user")
            db_name = os.getenv("db_name")
            host = os.getenv("host")
            engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db_name}")
            print("✅ Database connection successful!")
            return engine
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            exit(1)

    def fetch_data(self):
        """Fetches and preprocesses data from MySQL."""
        query = "SELECT * FROM Bookings;"
        df = pd.read_sql(query, self.engine)

        for col in ['travel_date', 'booking_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'booking_date' in df.columns and 'travel_date' in df.columns:
            df['days_between_booking_travel'] = (df['travel_date'] - df['booking_date']).dt.days
        else:
            df['days_between_booking_travel'] = None

        df = df.dropna(subset=['price'])
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def prepare_data_regression(self):
        features = ['user_id']
        if 'days_between_booking_travel' in self.df.columns:
            features.append('days_between_booking_travel')

        X = self.df[features].fillna(0)
        y = self.df['price'].fillna(0)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def prepare_data_classification(self):
        self.df['price_category'] = (self.df['price'] > self.df['price'].median()).astype(int)
        features = ['user_id']
        if 'days_between_booking_travel' in self.df.columns:
            features.append('days_between_booking_travel')

        X = self.df[features].fillna(0)
        y = self.df['price_category']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, model_name, model, is_classification=False):
        if is_classification:
            X_train, X_test, y_train, y_test = self.prepare_data_classification()
        else:
            X_train, X_test, y_train, y_test = self.prepare_data_regression()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            self.models[model_name] = {'pipeline': pipeline, 'accuracy': accuracy, 'report': report, 'y_test': y_test,
                                       'y_pred': y_pred}
            print(f"✅ {model_name} Accuracy: {accuracy:.2f}\n")
            print(report)

            results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred, 'Model': model_name})
            results_df.to_csv(f'{model_name}_classification_results.csv', index=False)
            print(f"✅ Classification results saved for {model_name} in '{model_name}_classification_results.csv'.")

            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Affordable', 'Expensive'],
                        yticklabels=['Affordable', 'Expensive'])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix: {model_name}")
            plt.show()
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.models[model_name] = {'pipeline': pipeline, 'mse': mse, 'r2': r2, 'y_test': y_test, 'y_pred': y_pred}
            print(f"✅ {model_name} Results:\nMSE: {mse:.2f}, R²: {r2:.2f}\n")

            results_df = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred, 'Model': model_name})
            results_df.to_csv(f'{model_name}_predictions.csv', index=False)
            print(f"✅ Predictions saved for {model_name} in '{model_name}_predictions.csv'.")

    def compare_models(self):
        for model_name, model_data in self.models.items():
            y_test = model_data['y_test']
            y_pred = model_data['y_pred']

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions',
                                     marker=dict(color='green', size=8, opacity=0.6)))
            fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines',
                                     name='Perfect Prediction', line=dict(color='red', dash='dash')))
            fig.update_layout(title=f'Actual vs Predicted Prices ({model_name})',
                              xaxis_title='Actual Price', yaxis_title='Predicted Price', template='plotly_dark')
            fig.show()

    def run_analysis(self):
        self.train_model("Linear Regression", LinearRegression())
        self.train_model("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        self.train_model("Logistic Regression", LogisticRegression(), is_classification=True)
        self.train_model("Random Forest Classifier", RandomForestClassifier(n_estimators=100, random_state=42),
                         is_classification=True)
        self.compare_models()


if __name__ == "__main__":
    analysis = TravelTechAnalysis()
    analysis.run_analysis()
