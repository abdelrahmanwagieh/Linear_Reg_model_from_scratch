

# Insurance Cost Prediction Project

## Overview

This project involves predicting insurance charges based on various features using both a custom-built linear regression model and a scikit-learn implementation. The dataset used for this project is from Kaggle and includes information on age, sex, BMI, number of children, smoking status, and region.

## Project Structure

- **`insurance.ipynb`**: The main Jupyter Notebook containing the entire workflow of the project.
- **`insurance.csv`**: The dataset used for analysis and modeling.

## Dataset

The dataset `insurance.csv` contains the following columns:
- `age`: Age of the individual.
- `sex`: Gender of the individual.
- `bmi`: Body Mass Index of the individual.
- `children`: Number of children/dependents.
- `smoker`: Smoking status.
- `region`: Region of residence.
- `charges`: Insurance charges (target variable).

## Steps and Workflow

1. **Data Download and Preparation**:
   - Downloaded the dataset from Kaggle and unzipped it.
   - Loaded the dataset into a DataFrame and performed initial exploration.
   - Removed duplicates and checked for missing values.

2. **Exploratory Data Analysis (EDA)**:
   - Described the data and visualized distributions for categorical and numeric columns.
   - Applied One-Hot Encoding to categorical variables for machine learning compatibility.
   - Plotted a correlation heatmap to visualize relationships between features.

3. **Custom Linear Regression Model**:
   - Implemented a linear regression model from scratch:
     - Split the data into training and testing sets.
     - Applied standard scaling and added a bias term.
     - Computed predictions and errors.
     - Implemented gradient descent to update model parameters.
     - Evaluated the model using R² score and cost function.

4. **Scikit-Learn Linear Regression Model**:
   - Trained a linear regression model using scikit-learn:
     - Standardized features and split the data.
     - Trained the model and made predictions.
     - Evaluated the model using R² score and root mean squared error (RMSE).

5. **Model Comparison**:
   - Compared the performance of the custom-built model with the scikit-learn model.
   - Plotted learning curves, residual plots, and comparison of predicted vs. actual charges.

## Results

- **Custom Model**:
  - Final cost with updated theta: `19,357,454.55`
  - R² Score: `0.734`

- **Scikit-Learn Model**:
  - R² Score: `0.809`

## Visualizations

- Distribution plots for categorical features.
- Histograms for numeric features.
- Correlation heatmap.
- Learning curve showing cost vs. number of iterations.
- Comparison plots between custom model and scikit-learn model predictions.

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdelrahmanwagieh/Linear_Reg_model_from_scratch.git
   ```
   
2. **Navigate to the project directory**:
   ```bash
   cd Linear_Reg_model_from_scratch

   ```

3. **Install dependencies** (if using a local environment):
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

4. **Run the notebook**:
   - Open `insurance.ipynb` in Jupyter Notebook or Google Colab.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Dataset: [Insurance Costs Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

Feel free to adjust the sections to better fit your project or provide more details as needed.
