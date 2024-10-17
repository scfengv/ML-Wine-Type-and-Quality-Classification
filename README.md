# Wine Classification Project

This project focuses on classifying wines based on their type (red or white) and quality using machine learning techniques. The data used in this project is sourced from the UC Irvine Machine Learning Repository's [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

## Project Structure

The project consists of two main Python scripts:

1. `Wine_type_classification.py`: Classifies wines as red (1) or white (0).
2. `Wine_quality_classification.py`: Classifies wines based on their quality (scores ranging from 3 to 9).

## Features

- Implements various machine learning models including Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Support Vector Classifier, Random Forest, and XGBoost.
- Utilizes grid search for hyperparameter tuning.
- Performs feature selection using Sequential Feature Selector.
- Includes data preprocessing techniques such as standard scaling for certain models.
- Generates performance metrics, ROC curves, confusion matrices, and decision regions (for wine type classification).
- Handles class imbalance in wine quality classification using SMOTE (Synthetic Minority Over-sampling Technique).

## Requirements

- Python 3.x
```python
pip install -r requirements.txt
```

## Usage

1. Ensure you have the required libraries installed.
2. Place the Wine.csv file in a `data` folder in the project directory.
3. Run the scripts:

   ```
   python Wine_type_classification.py
   python Wine_quality_classification.py
   ```

4. Results will be saved in the `result/Wine_Type` and `result/Wine_Quality` directories respectively.

## Output

The scripts generate the following outputs:

- Classification reports (CSV)
- ROC curves (PNG)
- Confusion matrices (PNG)
- Decision regions (PNG, only for wine type classification)
- Overall results summary (CSV)

## Notes

- The wine quality classification script includes options for running with original data or upsampled data to handle class imbalance.
- Some models (Logistic Regression, K-Nearest Neighbors, Support Vector Classifier) use standardized features.

## Future Improvements

- Implement cross-validation for more robust performance estimation.
- Explore ensemble methods combining multiple models.
- Experiment with deep learning approaches for potentially improved performance.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- UC Irvine Machine Learning Repository for providing the Wine Quality Dataset.
- The scikit-learn, XGBoost, and mlxtend communities for their excellent machine learning tools and libraries.
