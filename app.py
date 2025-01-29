from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from flask_mail import Mail, Message
from flask_wtf.csrf import CSRFProtect
from fpdf import FPDF
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sqlite3
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Secret key for flash messages and session management
app.secret_key = 'your_secret_key'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_email_password'  # Replace with your email password
mail = Mail(app)

# Enable CSRF protection
csrf = CSRFProtect(app)

# Model path
MODEL_PATH = 'model/breast_cancer_inceptionv3.h5'

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the model exists.")
model = load_model(MODEL_PATH)

# Database Initialization
def init_db():
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            symptoms TEXT,
            duration TEXT,
            contact TEXT,
            email TEXT,
            other_diseases TEXT,
            medications TEXT,
            report_link TEXT
        )
    ''')
    conn.close()

# Initialize the database
init_db()

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to send email with the report link or PDF
def send_email_with_report(recipient, report_link, pdf_path, patient_name):
    try:
        msg = Message(f"Medical Report for {patient_name}",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=[recipient])
        
        # Email body
        msg.body = (
            f"Dear {patient_name},\n\n"
            f"Your medical report is ready. Please find the details below:\n\n"
            f"Report Link: {report_link}\n\n"
            f"Alternatively, the attached PDF contains your complete report.\n\n"
            f"This is a system-generated email. Please do not reply.\n\n"
            f"Thank you for using our service.\n\n"
            f"Best Regards,\nBreast Cancer Detection Team"
        )
        
        # Attach the PDF report
        with app.open_resource(pdf_path) as pdf:
            msg.attach("patient_report.pdf", "application/pdf", pdf.read())

        mail.send(msg)
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to generate PDF report
def generate_pdf(result, filepath, patient_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Patient Report', ln=True, align='C')

    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Name: {patient_data['name']}", ln=True)
    pdf.cell(0, 10, f"Age: {patient_data['age']}", ln=True)
    pdf.cell(0, 10, f"Symptoms: {patient_data['symptoms']}", ln=True)
    pdf.cell(0, 10, f"Remark: {result['remark']}", ln=True)
    pdf.cell(0, 10, f"Prediction Class: {result['class']}", ln=True)
    pdf.cell(0, 10, f"Accuracy: {result['accuracy']}", ln=True)
    
    pdf.ln(10)
    pdf.cell(0, 10, "Tested Image:", ln=True)
    pdf.image(filepath, x=10, y=pdf.get_y(), w=100)

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'patient_report.pdf')
    pdf.output(pdf_path)
    return pdf_path

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact', methods=['GET', 'POST'])
@csrf.exempt
def contact():
    if request.method == 'POST':
        flash('Message sent successfully!')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Sample validation (replace with actual validation logic)
        if username == 'admin' and password == 'password':
            session['user'] = username
            return redirect(url_for('history'))
        else:
            flash('Invalid login credentials. Please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/history', methods=['GET', 'POST'])
def history():
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        symptoms = request.form.get('symptoms')
        duration = request.form.get('duration')
        contact = request.form.get('contact')
        email = request.form.get('email')
        other_diseases = request.form.get('other_diseases', '')
        medications = request.form.get('medications', '')

        if not name or not age or not symptoms or not duration or not contact or not email:
            flash('All required fields must be filled.')
            return redirect(url_for('history'))

        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patients (name, age, symptoms, duration, contact, email, other_diseases, medications)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, age, symptoms, duration, contact, email, other_diseases, medications))
        conn.commit()
        conn.close()

        flash('Patient details saved successfully.', 'success')
        return redirect(url_for('upload'))

    return render_template('history.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected. Please upload an image.')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Unsupported file format. Please upload a PNG, JPG, or JPEG image.')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict(filepath)

        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients ORDER BY id DESC LIMIT 1')
        patient = cursor.fetchone()
        conn.close()

        patient_data = {
            'name': patient[1],
            'age': patient[2],
            'symptoms': patient[3],
            'email': patient[6]
        }

        pdf_path = generate_pdf(result, filepath, patient_data)

        if patient_data['email']:
            report_link = f"http://127.0.0.1:5000/static/uploads/{filename}"
            send_email_with_report(patient_data['email'], report_link, pdf_path, patient_data['name'])

        flash('Report generated successfully.', 'success')
        return render_template('patient_report.html', result=result, filename=filename)

    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/download')
def download_report():
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'patient_report.pdf')
    return send_file(pdf_path, as_attachment=True)

def predict(filepath):
    try:
        image = load_img(filepath, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        prediction = model.predict(image)
        accuracy = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
        if prediction[0][0] > 0.5:
            return {
                'class': 'Malignant (Cancerous)',
                'accuracy': f"{accuracy:.2f}%",
                'remark': "Cancer detected. Consult a doctor immediately.",
            }
        else:
            return {
                'class': 'Benign (Non-Cancerous)',
                'accuracy': f"{accuracy:.2f}%",
                'remark': "No cancer detected.",
            }
    except Exception as e:
        flash(f"Error in prediction: {e}")
        return {'class': 'Error', 'accuracy': 'N/A', 'remark': 'An error occurred during prediction.'}

if __name__ == '__main__':
    app.run(debug=True)
