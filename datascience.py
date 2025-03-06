import pandas as pd
import plotly.express as pe
import plotly.graph_objects as go
import os
import dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

        # Convert dates and handle missing values
        for col in ['travel_date', 'booking_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'booking_date' in df.columns and 'travel_date' in df.columns:
            df['days_between_booking_travel'] = (df['travel_date'] - df['booking_date']).dt.days
        else:
            df['days_between_booking_travel'] = None  # Avoid KeyError

        # Drop any rows with missing target (price)
        df = df.dropna(subset=['price'])

        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def exploratory_data_analysis(self):
        """Performs basic exploratory data analysis and visualization."""
        print(self.df.describe())
        self.df.describe().to_csv('data_summary.csv')
        print("✅ Data summary saved to 'data_summary.csv'")

        # Histogram for price distribution
        fig = pe.histogram(self.df, x='price', nbins=30, marginal="box", title='Price Distribution of Bookings',
                           color_discrete_sequence=['blue'])
        fig.update_layout(xaxis_title='Price', yaxis_title='Count', template='plotly_dark', bargap=0.2)
        fig.show()

        # Price over time
        if 'travel_date' in self.df.columns:
            df_sorted = self.df.sort_values(by='travel_date')
            fig = pe.line(df_sorted, x='travel_date', y='price', title='Booking Prices Over Time', markers=True,
                          color_discrete_sequence=['red'])
            fig.update_layout(xaxis_title='Travel Date', yaxis_title='Price', template='plotly_dark')
            fig.show()

        # Impact of booking lead time on price
        if 'days_between_booking_travel' in self.df.columns and self.df['days_between_booking_travel'].notnull().all():
            fig = pe.box(self.df, x='days_between_booking_travel', y='price', title='Impact of Booking Time on Price',
                         color='days_between_booking_travel', color_continuous_scale='blues')
            fig.update_layout(xaxis_title='Days Between Booking and Travel', yaxis_title='Price',
                              template='plotly_dark')
            fig.show()

    def prepare_data(self):
        """Prepares features and target for model training."""
        features = ['user_id']
        if 'days_between_booking_travel' in self.df.columns:
            features.append('days_between_booking_travel')

        X = self.df[features].fillna(0) # Fill missing values with zero
        y = self.df['price'].fillna(0)

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, model_name, model):
        """Trains a machine learning model using a pipeline."""
        X_train, X_test, y_train, y_test = self.prepare_data()

        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            ('model', model)  # Apply model
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.models[model_name] = {
            'pipeline': pipeline,
            'mse': mse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }

        print(f"✅ {model_name} Results:\nMSE: {mse:.2f}, R²: {r2:.2f}\n")

        # Save predictions to CSV
        results_df = pd.DataFrame({
            'Actual Price': y_test.values,
            'Predicted Price': y_pred,
            'Model': model_name
        })
        results_df.to_csv(f'{model_name}_predictions.csv', index=False)
        print(f"✅ Predictions saved for {model_name} in '{model_name}_predictions.csv'.")

        # Save MSE and R2 score
        summary_df = pd.DataFrame({'Model': [model_name], 'MSE': [mse], 'R2': [r2]})
        summary_df.to_csv(f'{model_name}_summary.csv', index=False)
        print(f"✅ Model summary saved for {model_name} in '{model_name}_summary.csv'.")

    def compare_models(self):
        """Compares model predictions using scatter plots."""
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


if __name__ == "__main__":
    analysis = TravelTechAnalysis()
    analysis.exploratory_data_analysis()

    # Train models with pipeline
    analysis.train_model("Linear Regression", LinearRegression())
    analysis.train_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42))

    # Compare model performance
    analysis.compare_models()
