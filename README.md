# Twitter Sentiment Analysis Using Machine Learning: 

This project focuses on analyzing Twitter sentiments using machine learning techniques. Below is a comprehensive summary capturing each category and step involved in the analysis.

## Project Setup
1. Environment Setup:
   - Installed necessary packages including Kaggle for dataset download and Scikit-Learn for machine learning tasks.
   - Mounted Google Drive to access Kaggle credentials and store datasets.

## Data Acquisition
1. Dataset Download:
   - Downloaded the Sentiment140 dataset from Kaggle.
   - Extracted the dataset using the `zipfile` library.

## Data Loading and Initial Inspection
1. Loading the Data:
   - Loaded the CSV file into a pandas DataFrame.
   - Checked the number of rows and columns to understand the dataset size.
   - Displayed the first few rows to get an overview of the data.

## Data Preprocessing
1. Column Naming:
   - Renamed the columns for better understanding:
     ```python
     column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
     ```

2. Handling Missing Values:
   - Checked for missing values in the dataset and confirmed there were none.

3. Target Variable Distribution:
   - Analyzed the distribution of the target column to ensure an even distribution for model training.

4. Target Value Conversion:
   - Converted the target value `4` to `1` to represent positive tweets, making it binary:
     ```python
     Twitter_data.replace({4: 1}, inplace=True)
     ```

## Data Transformation
1. Text Preprocessing:
   - Stemming:
     - Applied stemming to reduce words to their root forms using the `PorterStemmer`:
       ```python
       def stemming(content):
           stemmed_content = re.sub('[^a-zA-Z]',' ', content).lower().split()
           stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
           return ' '.join(stemmed_content)
       Twitter_data['stemmed_content'] = Twitter_data['text'].apply(stemming)
       ```

## Feature Extraction
1. Text to Numeric Conversion:
   - Used `TfidfVectorizer` to convert textual data into numerical form for model training:
     ```python
     vectorizer = TfidfVectorizer()
     X_train = vectorizer.fit_transform(X_train)
     X_test = vectorizer.transform(X_test)
     ```

## Model Training and Evaluation
1. Model Training:
   - Trained a Logistic Regression model on the preprocessed data:
     ```python
     model = LogisticRegression(max_iter=1000)
     model.fit(X_train, Y_train)
     ```

2. Model Evaluation:
   - Evaluated the model using accuracy scores on both training and test datasets:
     ```python
     training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
     test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
     print('Accuracy score on the training data :', training_data_accuracy)
     print('Accuracy score on the test data :', test_data_accuracy)
     ```

## Model Deployment
1. Model Saving:
   - Saved the trained model using `pickle` for future use:
     ```python
     filename = 'trained_model.sav'
     pickle.dump(model, open(filename, 'wb'))
     ```

2. Loading and Using the Saved Model:
   - Loaded the saved model and made predictions on new data:
     ```python
     loaded_model = pickle.load(open('/content/trained_model.sav', 'rb'))
     prediction = loaded_model.predict(X_new)
     ```

## Results and Conclusion
1. Model Performance:
   - The logistic regression model achieved an accuracy score of approximately 77.8% on the test data.


