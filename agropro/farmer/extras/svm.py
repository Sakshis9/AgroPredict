import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class CropRecommenderSVM:
    def __init__(self, csv_path, train_model=True):
        self.csv_path = csv_path
        self.pipeline = None
        self.crop_encoder = None
        self.accuracy = None
        self.known_soil_types = None
        self.known_states = None
        self.known_seasons = None
        self._load_data()
        if train_model:
            self._prepare_model()

    def _load_data(self):
        self.data = pd.read_csv(self.csv_path)
        self.features = self.data[['Soil type', 'States', 'Season']]
        self.target_raw = self.data['Crops for mixed cropping']

    def _prepare_model(self):
        # Encode target crops
        self.crop_encoder = LabelEncoder()
        target = self.crop_encoder.fit_transform(self.target_raw)

        # OneHotEncode categorical columns
        column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(), ['Soil type', 'States', 'Season'])
            ]
        )

        # Build the pipeline
        self.pipeline = Pipeline([
            ('preprocessor', column_transformer),
            ('scaler', StandardScaler(with_mean=False)),  # For sparse matrix
            ('svm', SVC(kernel='linear', probability=True))
        ])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, target, test_size=0.2, random_state=42)

        # Train model
        self.pipeline.fit(X_train, y_train)

        # Save model accuracy
        self.accuracy = self.pipeline.score(X_test, y_test)
        print(f"✅ Crop Recommendation Model Accuracy: {self.accuracy:.2f}")

        # Classification report
        y_pred = self.pipeline.predict(X_test)
        print("📊 Classification Report:")
        print(classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(self.crop_encoder.classes_)),
            target_names=self.crop_encoder.classes_,
            zero_division=0
        ))

        # Save known categories for input validation
        self.known_soil_types = self.pipeline.named_steps['preprocessor'].transformers_[0][1].categories_[0]
        self.known_states = self.pipeline.named_steps['preprocessor'].transformers_[0][1].categories_[1]
        self.known_seasons = self.pipeline.named_steps['preprocessor'].transformers_[0][1].categories_[2]

    def recommend_crops(self, soil_type, state, season, use_exact_match=True):
        # Normalize input
        soil_type = soil_type.strip().capitalize()
        state = state.strip().title()
        season = season.strip().capitalize()

        # Try exact match
        if use_exact_match:
            exact_match = self.data[
                (self.data['Soil type'] == soil_type) &
                (self.data['States'] == state) &
                (self.data['Season'] == season)
            ]
            if not exact_match.empty:
                return exact_match['Crops for mixed cropping'].unique().tolist()

        # Validate input
        if (soil_type not in self.known_soil_types or
            state not in self.known_states or
            season not in self.known_seasons):
            return ["Error: Soil type, state, or season not recognized."]

        # Predict with model
        input_df = pd.DataFrame([[soil_type, state, season]], columns=['Soil type', 'States', 'Season'])
        prediction = self.pipeline.predict(input_df)
        return self.crop_encoder.inverse_transform(prediction).tolist()

    def save_model(self, model_path='svm_crop_model.pkl', encoder_path='label_encoder.pkl', accuracy_path='model_accuracy.txt'):
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.crop_encoder, encoder_path)

        # Save accuracy to text file
        if self.accuracy is not None:
            with open(accuracy_path, 'w') as f:
                f.write(f"Crop Recommender Model Accuracy: {self.accuracy:.4f}\n")

    def load_model(self, model_path='svm_crop_model.pkl', encoder_path='label_encoder.pkl'):
        self.pipeline = joblib.load(model_path)
        self.crop_encoder = joblib.load(encoder_path)
        self.known_soil_types = self.pipeline.named_steps['preprocessor'].transformers_[0][1].categories_[0]
        self.known_states = self.pipeline.named_steps['preprocessor'].transformers_[0][1].categories_[1]
        self.known_seasons = self.pipeline.named_steps['preprocessor'].transformers_[0][1].categories_[2]

# Optional test script
if __name__ == "__main__":
    csv_path = "/Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/uploads/CropRecommendation.csv"
    model_path = "/Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/extras/svm_crop_model.pkl"
    encoder_path = "/Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/extras/label_encoder.pkl"
    accuracy_path = "/Users/sakshi/Downloads/AgroPro_Final/agropro/farmer/extras/model_accuracy.txt"

    recommender = CropRecommenderSVM(csv_path)
    recommender.save_model(model_path, encoder_path, accuracy_path)

    # Test input
    test_soil = "Black"
    test_state = "Andhra Pradesh"
    test_season = "Rabi"

    prediction = recommender.recommend_crops(test_soil, test_state, test_season)
    print(f"🌿 Predicted crop(s): {prediction}")
