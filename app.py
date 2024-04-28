
import joblib
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# filename = 'diabetes-prediction-rfc-model.pkl'
# classifier = pickle.load(open(filename, 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))
# model1 = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


# Create database tables
# @app.before_first_request
# def create_tables():
    # db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/help')
def help():
    return render_template("help.html")


@app.route('/terms')
def terms():
    return render_template("tc.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:
            login_user(user, remember=form.remember.data)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
    return render_template("login.html", form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        new_user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully. Please login.', 'success')
        return redirect("/login")
    return render_template('signup.html', form=form)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    flash('An unexpected error occurred. Please try again later.', 'error')
    return redirect(url_for('index'))
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/disindex")

def disindex():
    return render_template("disindex.html")


@app.route("/cancer")
@login_required
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
@login_required
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")


# ---------------------------------------------------------------------------------------



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))



from flask import request, render_template

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    input_features = [int(x) for x in request.form.values()]
    
    # Create a DataFrame from the input features
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame([input_features], columns=features_name)
    
    # Load the trained breast cancer detection model
    breast_cancer_detection_model = pickle.load(open('breast_cancer_detection_model.pkl', 'rb'))
    
    # Make predictions using the loaded model
    prediction = breast_cancer_detection_model.predict(df)
    
    # Determine the result based on the prediction
    if prediction == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    # Render the template with the prediction result
    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))







#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('diab_result.html', prediction=my_prediction)


############################################################################################################


@app.route('/predictheart', methods=['POST'])
def predictheart():
    # Get input features from the form
    input_features = [float(x) for x in request.form.values()]
    
    # Define feature names
    features_name = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    # ["age", "trestbps", "chol", "thalach", "oldpeak", "sex_0",
    #                  "sex_1", "cp_0", "cp_1", "cp_2", "cp_3", "fbs_0",
    #                  "restecg_0", "restecg_1", "restecg_2", "exang_0", "exang_1",
    #                  "slope_0", "slope_1", "slope_2", "ca_0", "ca_1", "ca_2", "thal_1",
    #                  "thal_2", "thal_3"]
    
    
    # Create a dataframe with the input features
    df = pd.DataFrame([input_features], columns=features_name)
    
    # Load the trained model
    heart_attack_detection_model = pickle.load(open('heart_disease_prediction_model.pkl', 'rb'))
    
    # Predict the output
    output = heart_attack_detection_model.predict(df)

    # Determine the prediction result
    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    # Render the prediction result in a template
    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))


############################################################################################################


if __name__ == '__main__':
    # db.create_all()
    app.run(debug=True)

