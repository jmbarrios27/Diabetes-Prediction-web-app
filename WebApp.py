import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from PIL import Image
import streamlit as st
import seaborn as sns
# Create a title
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning
""")

# open and display an image in the web app
image = Image.open('D:\\pythonProject\\diabetesimage.png')
st.image(image, caption='ML', use_column_width=True)

# Get Data
data = pd.read_csv('D:\\pythonProject\\diabetes.csv', delimiter=',')
# Set a subheader

# Function to transform target variable


def target_change(Outcome):
    if Outcome == 1:
        return 'Diabetes'
    else:
        return 'No Diabetes'


# Creating a target variable with string values.
data['Target'] = data['Outcome'].apply(target_change)

st.subheader('Data Information: ')
# # Show data as a table
st.dataframe(data)

# # show statistics on the data
st.write(data.describe())

# Creating a new DF with the original values.
df = data.drop(columns=['Outcome'])

# Correlation
data_correlation = df.corr()
st.dataframe(data=data_correlation)

# create variables for scatterplot.
numeric_columns = df.select_dtypes(['int32', 'int64', 'float', 'float32', 'float64']).columns
select_box_one = st.sidebar.selectbox(label='X Axis for scatterplot', options=numeric_columns)
select_box_two = st.sidebar.selectbox(label='Y Axis for scatterplot', options=numeric_columns)

# create acutal scatterplot.
color = ['darkred', 'darkgreen']
scatter = sns.relplot(x=select_box_one, y=select_box_two, data=df, hue=df.Target, palette=color)
st.subheader('Relationship between variables')
st.pyplot(scatter)

# # Split data into independent and dep variables
X = data.iloc[:, 0:8].values
Y = data.iloc[:, -1].values

# Variable importance check.
attributes = data.drop(columns=['Target', 'Outcome'])
target_values = data['Outcome']
ExtraTreesClassifier = ExtraTreesClassifier()
ExtraTreesClassifier.fit(attributes, target_values)
feature_importance = pd.Series(ExtraTreesClassifier.feature_importances_, index=attributes.columns)
st.subheader('Feature Importance ExtraTreeClassifier')
st.bar_chart(data=feature_importance, use_container_width=True)

# # train and test  split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# User input.
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('bloodPressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('SkinThickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0 , 30.5)
    bmi = st.sidebar.slider('BMI', 0, 67, 32)
    dpf = st.sidebar.slider('DPF', 0, 2, 1)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'bmi': bmi,
                 'dpf': dpf,
                 'age': age
                 }
    # Transform data into dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display user input
st.subheader('User input: ')
st.write(user_input)


# Create and train the model.
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, y_train)

# # metrics
st.subheader('Model Test Accuracy score: ')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')

# Classification report.
st.subheader('Classification Report')
st.write(str(classification_report(y_test, RandomForestClassifier.predict(X_test))))

# # Store predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# # Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)

