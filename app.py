import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', static_url_path='')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = MODELS_DIR

# Load trained model and scaler
model = None
scaler = None
stats_data = None
patients_data = None

FEATURE_ORDER = [
    'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
    'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
    'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
    'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
    'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
    'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
]

def load_resources():
    global model, scaler, stats_data, patients_data
    try:
        model_path = os.path.join(MODELS_DIR, 'model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        stats_path = os.path.join(DATA_DIR, 'stats.json')
        patients_path = os.path.join(DATA_DIR, 'dataset.json')

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Successfully loaded model and scaler.")
        
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
                
        if os.path.exists(patients_path):
            with open(patients_path, 'r', encoding='utf-8') as f:
                patients_data = json.load(f)
                
    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

@app.route('/')
def index():
    return jsonify({
        'system': 'PulmoVision AI Backend API',
        'status': 'online',
        'frontend': 'Streamlit (run: streamlit run streamlit_app.py)',
        'endpoints': ['/api/health', '/api/stats', '/api/patients', '/api/predict']
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'patients_count': len(patients_data) if patients_data else 0
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    global stats_data
    stats_path = os.path.join(DATA_DIR, 'stats.json')
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
        except Exception:
            pass
    if stats_data:
        return jsonify(stats_data)
    return jsonify({'error': 'Stats not loaded'}), 500

@app.route('/api/patients', methods=['GET'])
def get_patients():
    if not patients_data:
        return jsonify({'error': 'Patient data not loaded'}), 500

    query = request.args.get('q', '').strip().lower()
    level = request.args.get('level', '').strip()
    gender = request.args.get('gender', '').strip()
    min_age = int(request.args.get('min_age', 0))
    max_age = int(request.args.get('max_age', 120))
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    sort_by = request.args.get('sort_by', 'Patient Id')
    sort_dir = request.args.get('sort_dir', 'asc')

    filtered = []
    for p in patients_data:
        if query and query not in str(p.get('Patient Id', '')).lower():
            continue
        if level and level != 'All' and p.get('Level') != level:
            continue
        if gender and gender != 'All':
            g_val = 1 if gender.lower() == 'male' else 2
            if p.get('Gender') != g_val:
                continue
        age = p.get('Age', 0)
        if not (min_age <= age <= max_age):
            continue
        filtered.append(p)

    # Sort
    def sort_key(item):
        val = item.get(sort_by, 0)
        if sort_by == 'Patient Id':
            # Extract number for natural sort if possible
            try:
                return int(str(val).replace('P', ''))
            except:
                return str(val)
        return val

    filtered.sort(key=sort_key, reverse=(sort_dir == 'desc'))

    total_filtered = len(filtered)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated = filtered[start_idx:end_idx]

    return jsonify({
        'total': total_filtered,
        'page': page,
        'limit': limit,
        'total_pages': max(1, int(np.ceil(total_filtered / limit))),
        'patients': paginated
    })

@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    if not patients_data:
        return jsonify({'error': 'Patient data not loaded'}), 500
    for p in patients_data:
        if str(p.get('Patient Id')).lower() == str(patient_id).lower():
            return jsonify(p)
    return jsonify({'error': 'Patient not found'}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        
        # Build feature vector in exact order
        input_values = []
        for feature in FEATURE_ORDER:
            val = data.get(feature)
            if val is None:
                # Default fallback based on population mean or typical baseline
                if feature == 'Gender':
                    val = 1
                elif feature == 'Age':
                    val = 35
                else:
                    val = 3
            input_values.append(float(val))

        input_arr = np.array([input_values])
        
        # Scale features using MinMaxScaler
        if scaler is not None and model is not None:
            scaled_arr = scaler.transform(input_arr)
            prediction = model.predict(scaled_arr)[0]
            probabilities_raw = model.predict_proba(scaled_arr)[0]
            classes = model.classes_
            
            prob_dict = {
                cls_name: round(float(prob) * 100, 1)
                for cls_name, prob in zip(classes, probabilities_raw)
            }
        else:
            # Rule-based fallback if model not loaded
            score = sum(input_values[2:]) / len(input_values[2:])
            if score > 5.2:
                prediction = 'High'
                prob_dict = {'High': 85.0, 'Medium': 12.0, 'Low': 3.0}
            elif score > 3.4:
                prediction = 'Medium'
                prob_dict = {'High': 15.0, 'Medium': 75.0, 'Low': 10.0}
            else:
                prediction = 'Low'
                prob_dict = {'High': 2.0, 'Medium': 18.0, 'Low': 80.0}

        # Calculate feature deviation / key risk contributors
        risk_drivers = []
        if stats_data and 'feature_stats' in stats_data:
            for feat in FEATURE_ORDER:
                if feat in ['Gender', 'Age']:
                    continue
                user_val = float(data.get(feat, 3))
                pop_mean = stats_data['feature_stats'][feat]['mean']
                max_val = stats_data['feature_stats'][feat]['max']
                
                # Severity ratio
                severity = (user_val / max_val) * 100
                if user_val >= 5 or user_val > (pop_mean + 1.2):
                    risk_drivers.append({
                        'feature': feat,
                        'user_value': user_val,
                        'pop_mean': pop_mean,
                        'max_value': max_val,
                        'severity_percent': round(severity, 1),
                        'status': 'Critical' if user_val >= 7 else 'Elevated'
                    })

        # Sort drivers by severity
        risk_drivers.sort(key=lambda x: x['severity_percent'], reverse=True)

        # Generate clinical recommendations
        recommendations = []
        if prediction == 'High':
            recommendations.append("Immediate comprehensive pulmonary diagnostic evaluation (Low-Dose CT Scan recommended).")
            recommendations.append("Urgent consultation with an Oncologist / Pulmonologist.")
            if data.get('Coughing of Blood', 0) >= 4:
                recommendations.append("Immediate bronchoscopy assessment for hemoptysis evaluation.")
            if data.get('Smoking', 0) >= 4 or data.get('Passive Smoker', 0) >= 4:
                recommendations.append("Immediate smoking cessation protocol with behavioral & pharmacological support.")
            if data.get('Air Pollution', 0) >= 5 or data.get('OccuPational Hazards', 0) >= 5:
                recommendations.append("Mitigate environmental toxins: utilize HEPA air filtration and certified N95/FFP3 respirators.")
        elif prediction == 'Medium':
            recommendations.append("Schedule follow-up chest imaging (X-Ray / Low-Dose CT) within 3-6 months.")
            recommendations.append("Pulmonary function testing (Spirometry) to assess airway obstruction.")
            if data.get('Smoking', 0) >= 3:
                recommendations.append("Enrolment in smoking cessation counseling.")
            if data.get('Balanced Diet', 7) <= 4:
                recommendations.append("Adopt antioxidant-rich diet with clinical nutritionist guidance.")
            recommendations.append("Routine symptom tracking (dry cough, wheezing, shortness of breath).")
        else:
            recommendations.append("Maintain low-risk status with regular annual health checkups.")
            recommendations.append("Continue non-smoking lifestyle and minimize passive smoke exposure.")
            recommendations.append("Maintain active aerobic exercise and balanced antioxidant-rich diet.")

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': prob_dict.get(prediction, 95.0),
            'risk_drivers': risk_drivers[:6],
            'recommendations': recommendations,
            'input_data': data
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting PulmoVision AI Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
