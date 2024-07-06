# Restaurant Delivery Time Prediction

## Project Overview

This project aims to predict the delivery time of restaurant orders based on various features such as restaurant ID, location, cuisines offered, average cost, minimum order amount, customer rating, votes, and reviews. The dataset is split into training and testing sets, and multiple machine learning models are evaluated to find the best predictor for delivery time.

### Real-Time Usage Scenarios

- **Food Delivery Services:** Optimize delivery times for various restaurants, enhancing customer satisfaction by providing accurate delivery time estimates.
- **Restaurant Management:** Help restaurant managers and owners better understand delivery performance and make data-driven decisions to improve efficiency.
- **Logistics and Operations:** Assist logistics teams in planning and optimizing delivery routes based on predicted delivery times.
- **Customer Support:** Enable customer support teams to provide more accurate information about delivery status and expected delivery times.
- **Marketing and Promotions:** Tailor marketing campaigns and promotions based on delivery performance and customer satisfaction metrics.

## Dataset Features

- **Restaurant:** A unique ID that represents a restaurant.
- **Location:** The location of the restaurant.
- **Cuisines:** The cuisines offered by the restaurant.
- **Average_Cost:** The average cost for one person/order.
- **Minimum_Order:** The minimum order amount.
- **Rating:** Customer rating for the restaurant.
- **Votes:** The total number of customer votes for the restaurant.
- **Reviews:** The number of customer reviews for the restaurant.
- **Delivery_Time:** The order delivery time of the restaurant (Target Variable).

## Project Structure

The project consists of the following steps:

1. **Data Loading:** Load the training and testing datasets.
2. **Data Exploration:** Examine the dataset structure, missing values, and summary statistics.
3. **Data Cleaning:** Clean and preprocess the data by handling missing values, converting data types, and extracting relevant features.
4. **Feature Engineering:** Split categorical features into multiple columns and encode them.
5. **Data Preprocessing:** Handle missing values, encode categorical variables, and scale the features.
6. **Modeling:** Train and evaluate different machine learning models, including Decision Trees, Random Forests, and Support Vector Machines.
7. **Prediction:** Use the best-performing model to predict delivery times on the test set and save the results.

## Installation

To run this project, you need the following libraries:

- pandas
- numpy
- re
- tqdm
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy tqdm seaborn scikit-learn
```

## Usage

1. **Load the Data:**
    ```python
    import pandas as pd

    train = pd.read_excel('path_to_your_train_file.xlsx')
    test = pd.read_excel('path_to_your_test_file.xlsx')
    ```

2. **Data Cleaning and Preprocessing:**
    - Clean `Average_Cost` and `Minimum_Order` columns.
    - Handle missing values in `Rating`, `Votes`, and `Reviews`.
    - Split `Location` and `Cuisines` into multiple features.
    - Encode categorical variables.
    - Scale numerical features.

3. **Model Training:**
    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    # Train a Decision Tree
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, Y_train)

    # Train a Random Forest
    Rforest = RandomForestClassifier()
    Rforest.fit(X_train, Y_train)

    # Train a Support Vector Machine
    svm_model = SVC(kernel='rbf', C=1, gamma="scale")
    svm_model.fit(X_train, Y_train)
    ```

4. **Model Evaluation:**
    ```python
    val_score = clf.score(X_Val, Y_Val)
    print("Decision Tree Validation Score:", val_score)

    val_score = Rforest.score(X_Val, Y_Val)
    print("Random Forest Validation Score:", val_score)

    val_score = svm_model.score(X_Val, Y_Val)
    print("SVM Validation Accuracy:", val_score)
    ```

5. **Make Predictions:**
    ```python
    Predictions = Rforest.predict(X_test)
    pd.DataFrame(Predictions, columns=['Delivery_Time']).to_excel('path_to_your_submission_file.xlsx', index=False)
    ```

## Results

The final model used for predictions is a Random Forest Classifier, which provided the best validation score. The predictions are saved in an Excel file.

## Conclusion

This project demonstrates the process of data cleaning, preprocessing, feature engineering, and model training to predict restaurant delivery times. The use of Decision Trees, Random Forests, and Support Vector Machines allows for a comparative analysis of model performance.

## Future Work

- Explore additional features that may improve prediction accuracy.
- Experiment with more advanced machine learning algorithms and ensemble methods.
- Implement cross-validation and hyperparameter tuning for better model optimization.

## Author

Mallikarjun Marthi

---

This README file provides a comprehensive overview of the project, including the dataset features, project structure, installation instructions, usage, results, and real-time usage scenarios. It aims to help other data scientists understand and replicate the work effectively.
