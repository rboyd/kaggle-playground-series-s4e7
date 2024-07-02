import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Combine the data for fitting the LabelEncoder
all_data = pd.concat([train_data, test_data])

# Initialize and fit the LabelEncoders
gender_encoder = LabelEncoder()
vehicle_age_encoder = LabelEncoder()
vehicle_damage_encoder = LabelEncoder()

all_data['Gender'] = gender_encoder.fit_transform(all_data['Gender'])
all_data['Vehicle_Age'] = vehicle_age_encoder.fit_transform(all_data['Vehicle_Age'])
all_data['Vehicle_Damage'] = vehicle_damage_encoder.fit_transform(all_data['Vehicle_Damage'])

# Apply transformations to train and test data
train_data['Gender'] = gender_encoder.transform(train_data['Gender'])
train_data['Vehicle_Age'] = vehicle_age_encoder.transform(train_data['Vehicle_Age'])
train_data['Vehicle_Damage'] = vehicle_damage_encoder.transform(train_data['Vehicle_Damage'])

test_data['Gender'] = gender_encoder.transform(test_data['Gender'])
test_data['Vehicle_Age'] = vehicle_age_encoder.transform(test_data['Vehicle_Age'])
test_data['Vehicle_Damage'] = vehicle_damage_encoder.transform(test_data['Vehicle_Damage'])

# Split data into X and y
X = train_data.drop(['id', 'Response'], axis=1)
y = train_data['Response']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate an XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')



# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")



# Prepare test data for prediction
X_test = test_data.drop('id', axis=1)

# Make predictions
test_data['Response'] = model.predict(X_test)

# Generate a timestamp
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

# Write predictions to a file
submission_file = f'submission-{timestamp}.csv'
test_data[['id', 'Response']].to_csv(submission_file, index=False)

print(f"Predictions written to {submission_file}")
