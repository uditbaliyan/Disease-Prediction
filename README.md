# RiskAssess


# Flask Prediction Web App

This Flask web application integrates three machine learning prediction models to provide health predictions for heart disease, diabetes, and breast cancer. The app uses the following models:
1. Heart Disease Prediction
2. Diabetes Prediction using Random Forest Classifier
3. Breast Cancer Detection

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
  - [Heart Disease Prediction](#heart-disease-prediction)
  - [Diabetes Prediction](#diabetes-prediction)
  - [Breast Cancer Detection](#breast-cancer-detection)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   flask run
   ```

The app will be accessible at `http://127.0.0.1:5000`.

## Usage

1. **Heart Disease Prediction:** Navigate to the Heart Disease Prediction page, fill out the form with the required medical information, and click "Predict" to get the prediction.
2. **Diabetes Prediction:** Navigate to the Diabetes Prediction page, fill out the form with the necessary details, and click "Predict" to get the prediction.
3. **Breast Cancer Detection:** Navigate to the Breast Cancer Detection page, provide the relevant data, and click "Predict" to get the prediction.



## Model Information

### Heart Disease Prediction
The heart disease prediction model uses various medical parameters such as age, sex, blood pressure, cholesterol levels, and more to predict the likelihood of a heart disease.

### Diabetes Prediction
The diabetes prediction model uses a Random Forest Classifier to predict the likelihood of diabetes based on factors like glucose levels, blood pressure, skin thickness, insulin levels, BMI, age, etc.

### Breast Cancer Detection
The breast cancer detection model classifies whether a tumor is benign or malignant based on several features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.


## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any changes you'd like to make.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

