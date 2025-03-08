import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('movies.csv', encoding='ISO-8859-1')  # Adjust encoding if needed

# Step 2: Data Cleaning
data.fillna({'Votes': 0}, inplace=True)  # Fill missing values for Votes

# Step 3: Clean the Duration Column
# Check for non-numeric values in the Duration column
print("Unique values in 'Duration' before cleaning:")
print(data['Duration'].unique())

# Remove non-numeric characters and convert to numeric
data['Duration'] = data['Duration'].str.extract('(\d+)')[0]  # Extract numeric part
data['Duration'] = pd.to_numeric(data['Duration'].str.replace(',', ''), errors='coerce')  # Remove commas and convert

# Check for NaN values in the Duration column
if data['Duration'].isnull().any():
    print("NaN values found in 'Duration' column. Filling with the median duration.")
    median_duration = data['Duration'].median()  # Calculate the median duration
    data['Duration'] = data['Duration'].fillna(median_duration)  # Fill NaN values with the median

# Step 4: Clean the Votes Column
# Convert all values to strings and clean
def clean_votes(vote):
    if isinstance(vote, str):  # Check if the value is a string
        if 'M' in vote:
            return float(vote.replace('$', '').replace('M', '').replace(',', '').strip()) * 1_000_000
        elif 'K' in vote:
            return float(vote.replace('$', '').replace('K', '').replace(',', '').strip()) * 1_000
        else:
            return float(vote.replace('$', '').replace(',', '').strip())
    return float(vote)  # Convert directly if it's already a number

# Convert Votes to string and clean the Votes column
data['Votes'] = data['Votes'].astype(str).apply(clean_votes)  # Clean the Votes column

# Step 5: Drop the Year Column (if it exists)
if 'Year' in data.columns:
    data.drop('Year', axis=1, inplace=True)  # Drop the Year column

# Step 6: Feature Encoding
data = pd.get_dummies(data, columns=['Genre', 'Director'], drop_first=True)

# Step 7: Define Features and Target
X = data.drop(['Name', 'Rating'], axis=1)  # Features
y = data['Rating']  # Target variable

# Step 8: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Step 11: Function to Predict Rating Based on User Input
def predict_rating():
    print("Enter the following details to predict the movie rating:")
    
    # User inputs
    duration = int(input("Duration in minutes: "))
    votes = int(input("Number of Votes: "))
    genre = input("Genre (comma-separated if multiple): ")
    director = input("Director: ")
    actor1 = input("Actor 1: ")
    
    # Create a DataFrame for the input
    input_data = {
        'Duration': duration,
        'Votes': votes,
        'Genre': genre,
        'Director': director,
        'Actor 1': actor1
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    input_df = pd.get_dummies(input_df, columns=['Genre', 'Director'], drop_first=True)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Align with training data
    
    # Make prediction
    predicted_rating = model.predict(input_df)
    print(f"Predicted Rating: {predicted_rating[0]:.2f}")

# Call the prediction function
predict_rating()
