import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
* Please run the command --> " streamlit run Frontend_UI/stroke_ui.py " in the terminal
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class StrokePredictionUI:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.dataset_path = r"C:\Users\HP\Documents\GitHub\ci-coursework\Data\cleaned-stroke-prediction-dataset-balanced.csv"

    def load_and_preprocess_data(self):
        """Load and preprocess the training dataset"""
        # Load dataset
        df = pd.read_csv(self.dataset_path)

        # Remove 'id' column
        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        # Split features and target
        X = df.drop('stroke', axis=1)
        y = df['stroke']

        # Scale numerical features only
        numerical_columns = ['age', 'avg_glucose_level', 'bmi']
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        return X, y

    def create_and_train_model(self, hyperparameters):
        """Create and train model with the given hyperparameters"""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create model
        model = Sequential([
            Dense(int(hyperparameters['Layer 1 Neurons']), activation='relu',
                  kernel_regularizer=l2(hyperparameters['L2 Regularization']),
                  input_shape=(X_train.shape[1],)),
            Dropout(hyperparameters['Dropout Rate 1']),
            Dense(int(hyperparameters['Layer 2 Neurons']), activation='relu',
                  kernel_regularizer=l2(hyperparameters['L2 Regularization'])),
            Dropout(hyperparameters['Dropout Rate 2']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['Learning Rate']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train model
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=20,
                            batch_size=32,
                            verbose=1)

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        return model, history, test_accuracy

    def preprocess_input(self, input_data):
        """Preprocess the user input data"""
        # Create a DataFrame with all possible columns, initialized to 0
        processed_data = pd.DataFrame(0, index=[0], columns=[
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'avg_glucose_level', 'bmi',
            'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban',
            'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
        ])

        # Fill in the basic fields
        processed_data['gender'] = 1 if input_data['gender'] == 'Female' else 0
        processed_data['age'] = input_data['age']
        processed_data['hypertension'] = 1 if input_data['hypertension'] == 'Yes' else 0
        processed_data['heart_disease'] = 1 if input_data['heart_disease'] == 'Yes' else 0
        processed_data['ever_married'] = 1 if input_data['ever_married'] == 'Yes' else 0
        processed_data['avg_glucose_level'] = input_data['avg_glucose_level']
        processed_data['bmi'] = input_data['bmi']

        # Set the appropriate work_type column to 1
        work_type_col = f"work_type_{input_data['work_type']}"
        if work_type_col in processed_data.columns:
            processed_data[work_type_col] = 1

        # Set the residence type
        if input_data['residence_type'] == 'Urban':
            processed_data['Residence_type_Urban'] = 1

        # Set the smoking status
        smoking_status_col = f"smoking_status_{input_data['smoking_status']}"
        if smoking_status_col in processed_data.columns:
            processed_data[smoking_status_col] = 1

        # Scale numerical features
        numerical_columns = ['age', 'avg_glucose_level', 'bmi']
        processed_data[numerical_columns] = self.scaler.transform(processed_data[numerical_columns])

        return processed_data

    def render_ui(self):
        st.title('Stroke Prediction Model using Artificial Neural Networks')

        # Hyperparameters input section
        st.header('Model Hyperparameters')
        st.write('Enter the Optimized Hyperparameters received:')

        col1, col2 = st.columns(2)
        with col1:
            layer1_neurons = st.number_input('Layer 1 Neurons', min_value=1, value=64)
            dropout_rate1 = st.number_input('Dropout Rate 1', min_value=0.0, max_value=1.0, value=0.2)
            learning_rate = st.number_input('Learning Rate', min_value=0.0001, max_value=0.1, value=0.001, format='%f')

        with col2:
            layer2_neurons = st.number_input('Layer 2 Neurons', min_value=1, value=32)
            dropout_rate2 = st.number_input('Dropout Rate 2', min_value=0.0, max_value=1.0, value=0.2)
            l2_reg = st.number_input('L2 Regularization', min_value=0.0, max_value=0.1, value=0.01, format='%f')

        # Patient details input section
        st.header('Patient Details')

        col3, col4 = st.columns(2)
        with col3:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            age = st.number_input('Age', min_value=0, max_value=150, value=30)
            hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
            heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
            ever_married = st.selectbox('Ever Married', ['No', 'Yes'])

        with col4:
            work_type = st.selectbox('Work Type', ['Never_worked', 'Private',
                                                   'Self-employed', 'children'])
            residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
            avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=100.0)
            bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, value=20.0)
            smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked',
                                                             'smokes'])

        # Prediction button
        if st.button('Train Model and Predict'):
            try:
                with st.spinner('Training model with provided hyperparameters...'):
                    # Create hyperparameters dictionary
                    hyperparameters = {
                        'Layer 1 Neurons': layer1_neurons,
                        'Layer 2 Neurons': layer2_neurons,
                        'Dropout Rate 1': dropout_rate1,
                        'Dropout Rate 2': dropout_rate2,
                        'Learning Rate': learning_rate,
                        'L2 Regularization': l2_reg
                    }

                    # Create input data dictionary
                    input_data = {
                        'gender': gender,
                        'age': age,
                        'hypertension': hypertension,
                        'heart_disease': heart_disease,
                        'ever_married': ever_married,
                        'work_type': work_type,
                        'residence_type': residence_type,
                        'avg_glucose_level': avg_glucose_level,
                        'bmi': bmi,
                        'smoking_status': smoking_status
                    }

                    # Train model
                    self.model, history, test_accuracy = self.create_and_train_model(hyperparameters)

                    # Display training results
                    st.subheader('Model Training Results')
                    st.write(f'Test Accuracy: {test_accuracy:.2%}')

                    # Plot training history
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        st.line_chart(pd.DataFrame(history.history['loss'], columns=['Training Loss']))
                    with fig_col2:
                        st.line_chart(pd.DataFrame(history.history['accuracy'], columns=['Training Accuracy']))

                    # Preprocess input and make prediction
                    preprocessed_input = self.preprocess_input(input_data)
                    prediction = self.model.predict(preprocessed_input)[0][0]

                    # Display prediction results
                    st.header('Prediction Results')
                    risk_percentage = prediction * 100

                    if risk_percentage < 50:
                        st.success(f'Low stroke risk')
                    else:
                        st.error(f'High stroke risk')

                    st.info("""
                    Note: This prediction is based on the provided information and should be used only as a general guide. 
                    Please consult with a healthcare professional for proper medical advice and diagnosis.
                    """)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check the error message and try again.")


def main():
    prediction_ui = StrokePredictionUI()
    prediction_ui.render_ui()


if __name__ == "__main__":
    main()

