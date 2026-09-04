# MLOps Tools & Technologies: Complete Guide with Indian Examples

## Table of Contents
1. [Machine Learning](#machine-learning)
2. [Version Control](#version-control)
3. [Streamlit](#streamlit)
4. [API](#api)
5. [Load Balancing & Uptime](#load-balancing)
6. [ECR (Elastic Container Registry)](#ecr)
7. [ECS (Elastic Container Service)](#ecs)
8. [Kubernetes](#kubernetes)
9. [Fargate](#fargate)
10. [EC2](#ec2)
11. [Deployment](#deployment)
12. [CI/CD](#cicd)
13. [Jenkins](#jenkins)
14. [MLflow](#mlflow)
15. [SageMaker](#sagemaker)
16. [Google Cloud Bucket](#gcs)

---

## 1. Machine Learning {#machine-learning}

### What is Machine Learning?

Machine Learning is teaching computers to learn from examples instead of being explicitly programmed. It's like teaching a child to recognize animals - you don't give rules, you show examples.

**Simple Analogy**: Imagine teaching someone to identify Indian sweets:
- **Traditional Programming**: "If it's round and orange, it's a ladoo. If it's diamond-shaped and silver, it's kaju katli."
- **Machine Learning**: Show 1000 photos of different sweets, and the computer learns to identify them itself!

### Why We Need Machine Learning?

1. **Pattern Recognition**: Identify spam SMS in Hindi/English mixed text
2. **Predictions**: Predict IPL match winners based on past data
3. **Personalization**: Netflix suggesting Bollywood movies you'll like
4. **Automation**: Auto-tagging faces in family WhatsApp photos

### Where It Stands in the Pipeline?

Machine Learning is the **CORE** of MLOps - everything else exists to support ML models:
```
Data → [MACHINE LEARNING] → Model → Deployment → Monitoring
```

### Real-World Alternatives?

1. **Rule-Based Systems**: Hard-coded if-else conditions
2. **Statistical Models**: Simple regression without learning
3. **Human Experts**: Manual decision making

### Code Example: IPL Winner Prediction

```python
# Traditional Approach (Rule-based)
def predict_ipl_winner_traditional(team1, team2, venue):
    if venue == "Wankhede" and team1 == "Mumbai Indians":
        return "Mumbai Indians"  # Home advantage
    elif team2 == "Chennai Super Kings" and venue == "Chepauk":
        return "Chennai Super Kings"
    else:
        return "Toss Winner"  # Random guess!

# Machine Learning Approach
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load IPL historical data
ipl_data = pd.DataFrame({
    'team1_wins_last_5': [3, 4, 2, 5, 1],
    'team2_wins_last_5': [2, 1, 4, 3, 4],
    'venue_advantage': [1, 0, 0, 1, 0],
    'toss_winner': [1, 0, 1, 1, 0],
    'winner': ['team1', 'team2', 'team2', 'team1', 'team2']
})

# Train model
features = ['team1_wins_last_5', 'team2_wins_last_5', 'venue_advantage', 'toss_winner']
X = ipl_data[features]
y = ipl_data['winner']

model = RandomForestClassifier()
model.fit(X, y)

# Predict
def predict_ipl_winner_ml(team1_form, team2_form, venue_adv, toss):
    prediction = model.predict([[team1_form, team2_form, venue_adv, toss]])
    confidence = model.predict_proba([[team1_form, team2_form, venue_adv, toss]]).max()
    return f"Winner: {prediction[0]} (Confidence: {confidence:.2%})"

# Example: MI vs CSK at Wankhede
print(predict_ipl_winner_ml(4, 3, 1, 1))  # MI in good form, home advantage, won toss
```

---

## 2. Version Control {#version-control}

### What is Version Control?

Version Control is like a time machine for your code and models. It tracks every change, who made it, and when.

**Simple Analogy**: Think of it as WhatsApp's "Message Edit History" but for code:
- See what changed
- Who changed it
- When it was changed
- Go back to any previous version

### Why We Need Version Control?

**Real Scenario**: Your team is building a UPI fraud detection model:
- Rahul improves accuracy to 95%
- Priya accidentally breaks the code
- Without version control: 😱 All work lost!
- With version control: 😌 Revert to Rahul's version in seconds

### Where It Stands in the Pipeline?

Version control is the **FOUNDATION** - it tracks everything:
```
Code Changes → Version Control → CI/CD → Deployment
Data Changes → Version Control → Training
Model Changes → Version Control → Model Registry
```

### Real-World Alternatives?

1. **Manual Backups**: Copy-paste folders (fraud_model_v1, fraud_model_v2_final, fraud_model_v2_final_FINAL)
2. **Email/Drive**: Sharing code via email or Google Drive
3. **No System**: YOLO approach (not recommended!)

### Code Example: Tracking Aadhaar Verification Model Changes

```bash
# Initialize Git repository for your Aadhaar verification project
git init aadhaar-verification-ml

# Track your first model
git add aadhaar_model.py
git commit -m "Initial Aadhaar face verification model - 85% accuracy"

# Priya improves the model
git add aadhaar_model.py
git commit -m "Added age detection feature - accuracy now 89%"

# Rahul adds fraud detection
git add fraud_detection.py
git commit -m "Added synthetic Aadhaar detection - catches 95% fake IDs"

# View history
git log --oneline
# Output:
# 3a4f5b6 Added synthetic Aadhaar detection - catches 95% fake IDs
# 2b3c4d5 Added age detection feature - accuracy now 89%
# 1a2b3c4 Initial Aadhaar face verification model - 85% accuracy

# Someone broke the code? No problem!
git revert 3a4f5b6  # Go back to Priya's stable version

# Working on Diwali discount feature? Create a branch
git checkout -b diwali-special-verification
# Make changes without affecting main code
```

**Advanced ML Version Control with DVC (Data Version Control):**

```python
# track_model_versions.py
import dvc.api
import mlflow

# Version control for large model files
def save_model_version(model, metrics, description):
    """
    Save model with version control
    Like saving different versions of Dosa recipes!
    """
    # Save model file
    model_path = f"models/upi_fraud_model_{metrics['accuracy']}.pkl"
    joblib.dump(model, model_path)
    
    # Track with DVC (handles large files)
    os.system(f"dvc add {model_path}")
    os.system(f"git add {model_path}.dvc")
    os.system(f'git commit -m "{description}"')
    
    # Push to remote storage (like S3)
    os.system("dvc push")
    
    return model_path

# Example usage
save_model_version(
    model=fraud_model,
    metrics={'accuracy': 0.96, 'precision': 0.94},
    description="UPI fraud model: Added Paytm transaction patterns"
)
```

---

## 3. Streamlit {#streamlit}

### What is Streamlit?

Streamlit is like PowerPoint for ML models - it turns Python scripts into beautiful web apps in minutes. No HTML/CSS/JavaScript needed!

**Simple Analogy**: 
- Building a web app traditionally: Like constructing a house brick by brick
- Using Streamlit: Like assembling a prefab home - just put pieces together!

### Why We Need Streamlit?

**Problem**: Your manager says "Show me how the crop prediction model works"
- **Without Streamlit**: Spend weeks building a web interface
- **With Streamlit**: Create a demo in 30 minutes!

### Where It Stands in the Pipeline?

Streamlit fits in the **DEMONSTRATION & PROTOTYPE** phase:
```
Model Development → Streamlit Demo → Stakeholder Approval → Production Deployment
                          ↓
                   Quick Feedback Loop
```

### Real-World Alternatives?

1. **Flask/Django**: More control but more complex
2. **Jupyter Notebooks**: Good for technical audience only
3. **Gradio**: Similar but less customizable
4. **Dash**: More complex, better for dashboards

### Code Example: Indian Crop Yield Predictor

```python
# crop_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# App title with emoji
st.title("🌾 Indian Crop Yield Predictor")
st.write("Helping farmers predict crop yields across India!")

# Sidebar for inputs
st.sidebar.header("Enter Farm Details")

# Input fields
state = st.sidebar.selectbox(
    "Select State",
    ["Maharashtra", "Punjab", "Uttar Pradesh", "West Bengal", "Gujarat"]
)

crop = st.sidebar.selectbox(
    "Select Crop",
    ["Rice", "Wheat", "Cotton", "Sugarcane", "Pulses"]
)

rainfall = st.sidebar.slider(
    "Annual Rainfall (mm)",
    min_value=200, max_value=3000, value=1000,
    help="Average rainfall in your area"
)

temperature = st.sidebar.slider(
    "Average Temperature (°C)",
    min_value=15, max_value=40, value=25
)

farm_size = st.sidebar.number_input(
    "Farm Size (hectares)",
    min_value=0.5, max_value=100.0, value=2.0
)

# Create model (in real app, load pre-trained model)
@st.cache_data  # Cache to avoid retraining
def get_model():
    # Simulated training data
    np.random.seed(42)
    X = np.random.rand(1000, 4) * [3000, 25, 50, 10]
    y = X[:, 0] * 0.01 + X[:, 1] * 2 + X[:, 2] * 0.5 + np.random.rand(1000) * 10
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# Predict button
if st.sidebar.button("🚀 Predict Yield"):
    model = get_model()
    
    # Create feature vector
    features = np.array([[rainfall, temperature, farm_size, 5]])  # 5 is encoded crop type
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Yield",
            value=f"{prediction:.1f} tons/hectare",
            delta=f"{prediction - 25:.1f} vs average"
        )
    
    with col2:
        st.metric(
            label="Revenue Estimate",
            value=f"₹{prediction * 50000:.0f}",
            delta="+12% vs last year"
        )
    
    with col3:
        st.metric(
            label="Confidence",
            value="85%",
            delta="High"
        )
    
    # Visualization
    st.subheader("📊 Yield Comparison")
    
    # Create comparison data
    comparison_data = pd.DataFrame({
        'Category': ['Your Prediction', 'District Average', 'State Average', 'National Average'],
        'Yield': [prediction, 25, 22, 20],
        'Color': ['green', 'blue', 'orange', 'red']
    })
    
    fig = px.bar(
        comparison_data, 
        x='Category', 
        y='Yield',
        color='Color',
        title=f"{crop} Yield Comparison",
        color_discrete_map={'green': '#2ecc71', 'blue': '#3498db', 'orange': '#f39c12', 'red': '#e74c3c'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("🌱 Recommendations")
    
    if rainfall < 800:
        st.warning("⚠️ Low rainfall detected. Consider:")
        st.write("- Install drip irrigation system")
        st.write("- Use drought-resistant seed varieties")
        st.write("- Apply for PM Krishi Sinchai Yojana")
    
    if prediction > 30:
        st.success("🎉 Excellent yield predicted!")
        st.write("- Consider organic certification for premium prices")
        st.write("- Connect with FPOs for better market access")
    
    # Historical trends
    st.subheader("📈 Historical Yield Trends")
    
    # Simulated historical data
    years = list(range(2018, 2024))
    historical_yields = [18, 20, 22, 24, 26, prediction]
    
    trend_data = pd.DataFrame({
        'Year': years,
        'Yield': historical_yields
    })
    
    fig2 = px.line(
        trend_data,
        x='Year',
        y='Yield',
        markers=True,
        title=f"{crop} Yield Trend in {state}"
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Information section
with st.expander("ℹ️ How it works"):
    st.write("""
    This app uses Machine Learning to predict crop yields based on:
    - **Historical data** from Indian Meteorological Department
    - **Soil data** from Soil Health Card scheme
    - **Crop patterns** from Agriculture Ministry
    - **Satellite imagery** from ISRO
    
    The model is trained on data from over 10,000 farms across India!
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Indian Farmers | Data source: Ministry of Agriculture")

# To run this app:
# streamlit run crop_predictor_app.py
```

This creates a beautiful, interactive web app that:
- Takes farmer inputs
- Predicts crop yield
- Shows revenue estimates
- Provides personalized recommendations
- Displays historical trends
- Works on mobile phones too!

All this with just Python - no web development knowledge needed! 

---

## 4. API (Application Programming Interface) {#api}

### What is an API?

API is like a waiter in a restaurant - it takes your order (request) to the kitchen (server) and brings back your food (response).

**Simple Analogy**: 
- **Swiggy App**: You (customer) → API (delivery partner) → Restaurant (server)
- **ML Model API**: Your app → API → ML Model → Prediction

### Why We Need APIs?

**Real Scenario**: Flipkart wants to use your price prediction model:
- **Without API**: Send them your code, data, and instructions (messy!)
- **With API**: Give them a URL - they send product details, get price back

### Where It Stands in the Pipeline?

API is the **BRIDGE** between your model and the world:
```
Trained Model → Model Serving → [API] → Applications/Users
                                  ↓
                          Request: "Is this UPI transaction fraud?"
                          Response: "Yes, 87% probability"
```

### Real-World Alternatives?

1. **Direct Database Access**: Risky and slow
2. **File Sharing**: Manual and not real-time
3. **Embedded Models**: Put model in every app (inefficient)
4. **RPC/gRPC**: More complex but faster

### Code Example: Aadhaar Verification API

```python
# aadhaar_verification_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from datetime import datetime
import cv2
import base64
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Aadhaar Verification API", version="1.0")

# Request/Response Models (like menu items)
class AadhaarRequest(BaseModel):
    aadhaar_number: str  # Masked: XXXX-XXXX-1234
    photo_base64: str    # Selfie photo
    name: str
    dob: str            # Date of birth
    
class AadhaarResponse(BaseModel):
    verified: bool
    confidence: float
    match_details: dict
    timestamp: str
    transaction_id: str

# Health check endpoint (like "Is restaurant open?")
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Aadhaar Verification API",
        "uptime": "99.9%",
        "response_time": "< 200ms"
    }

# Main verification endpoint
@app.post("/verify/aadhaar", response_model=AadhaarResponse)
async def verify_aadhaar(request: AadhaarRequest):
    """
    Verify Aadhaar details with photo matching
    Used by banks, telecom, and government services
    """
    try:
        # Step 1: Validate Aadhaar format
        if not validate_aadhaar_format(request.aadhaar_number):
            raise HTTPException(status_code=400, detail="Invalid Aadhaar format")
        
        # Step 2: Decode and process photo
        photo_bytes = base64.b64decode(request.photo_base64)
        photo_array = np.frombuffer(photo_bytes, np.uint8)
        photo = cv2.imdecode(photo_array, cv2.IMREAD_COLOR)
        
        # Step 3: Run face matching (simplified)
        face_match_score = perform_face_matching(photo)
        
        # Step 4: Verify other details
        name_match = verify_name(request.name)
        dob_match = verify_dob(request.dob)
        
        # Step 5: Calculate overall confidence
        confidence = calculate_confidence(face_match_score, name_match, dob_match)
        
        # Step 6: Prepare response
        return AadhaarResponse(
            verified=confidence > 0.85,
            confidence=confidence,
            match_details={
                "face_match": face_match_score,
                "name_match": name_match,
                "dob_match": dob_match
            },
            timestamp=datetime.now().isoformat(),
            transaction_id=generate_transaction_id()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# Bulk verification endpoint (for government schemes)
@app.post("/verify/bulk")
async def bulk_verify(aadhaar_list: list[AadhaarRequest]):
    """
    Verify multiple Aadhaar cards at once
    Used for PM Kisan, scholarship distribution, etc.
    """
    results = []
    for aadhaar in aadhaar_list:
        result = await verify_aadhaar(aadhaar)
        results.append(result)
    
    return {
        "total": len(aadhaar_list),
        "verified": sum(1 for r in results if r.verified),
        "failed": sum(1 for r in results if not r.verified),
        "results": results
    }

# Usage statistics endpoint
@app.get("/stats")
def get_stats():
    """
    API usage statistics for monitoring
    """
    return {
        "total_requests_today": 1_234_567,
        "success_rate": "98.5%",
        "average_response_time": "187ms",
        "peak_hour": "10:00-11:00 AM",
        "top_users": ["SBI", "Jio", "Airtel", "HDFC", "Paytm"]
    }

# Helper functions
def validate_aadhaar_format(aadhaar: str) -> bool:
    # Check format: XXXX-XXXX-1234
    return len(aadhaar.replace("-", "")) == 12

def perform_face_matching(photo: np.ndarray) -> float:
    # Simplified face matching
    # In production, use deep learning models
    return np.random.uniform(0.85, 0.99)

def verify_name(name: str) -> float:
    # Check against database
    return 0.95 if name else 0.0

def verify_dob(dob: str) -> float:
    # Verify date format and check database
    return 0.98 if dob else 0.0

def calculate_confidence(face: float, name: float, dob: float) -> float:
    # Weighted average
    return face * 0.6 + name * 0.2 + dob * 0.2

def generate_transaction_id() -> str:
    return f"UIDAI-{datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(1000, 9999)}"

# Rate limiting example
from fastapi import Request
from collections import defaultdict
import time

request_counts = defaultdict(lambda: {"count": 0, "window_start": time.time()})

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting: 100 requests per minute per API key
    Like Jio's daily data limit!
    """
    client_ip = request.client.host
    current_time = time.time()
    
    # Reset window if needed
    if current_time - request_counts[client_ip]["window_start"] > 60:
        request_counts[client_ip] = {"count": 0, "window_start": current_time}
    
    # Check limit
    if request_counts[client_ip]["count"] >= 100:
        return HTTPException(status_code=429, detail="Rate limit exceeded. Try after 1 minute.")
    
    request_counts[client_ip]["count"] += 1
    response = await call_next(request)
    return response

# To run this API:
# uvicorn aadhaar_verification_api:app --reload --port 8000

# Client code example
"""
import requests

# Using the API (like ordering from Swiggy)
response = requests.post(
    "http://localhost:8000/verify/aadhaar",
    json={
        "aadhaar_number": "XXXX-XXXX-1234",
        "photo_base64": "base64_encoded_photo_here",
        "name": "Raj Kumar",
        "dob": "01-01-1990"
    }
)

if response.json()["verified"]:
    print("✅ Aadhaar verified successfully!")
else:
    print("❌ Verification failed")
"""
```

---

## 5. Load Balancing & Uptime {#load-balancing}

### What is Load Balancing?

Load Balancing is like managing queues at a busy Indian railway ticket counter - distribute customers to multiple counters so no one counter gets overwhelmed.

**Simple Analogy**:
- **During Diwali Sale**: Flipkart has millions of users
- **Without Load Balancer**: One server handles all → 💥 Crash!
- **With Load Balancer**: Distributes users to 100 servers → 😊 Smooth shopping

### What is Uptime?

Uptime is how long your service stays available. Like how IRCTC should work 24/7 for ticket booking.
- **99% uptime** = Down 3.65 days/year (Bad for critical services)
- **99.9% uptime** = Down 8.76 hours/year (Acceptable)
- **99.99% uptime** = Down 52 minutes/year (Good!)
- **99.999% uptime** = Down 5 minutes/year (Excellent!)

### Why We Need Load Balancing & High Uptime?

**Real Scenario**: Cowin Vaccine Portal
- **Without Load Balancing**: Site crashed when millions tried to book slots
- **With Load Balancing**: Could handle 1 crore+ requests per minute

**Cost of Downtime**:
- **Paytm down for 1 hour** = ₹100+ crore loss
- **IRCTC down during Tatkal** = Thousands of missed journeys
- **Your ML Model API down** = Business decisions delayed

### Where It Stands in the Pipeline?

Load Balancing sits at the **ENTRY POINT**:
```
Users → [Load Balancer] → Server 1 (ML Model)
                       → Server 2 (ML Model)  
                       → Server 3 (ML Model)
                       
        Health Checks ← Monitoring
```

### Real-World Alternatives?

1. **Single Server**: Simple but risky
2. **Manual Switching**: Have backup server ready (slow)
3. **DNS Round Robin**: Basic distribution (no health checks)
4. **CDN**: For static content only

### Code Example: Load Balancer for IPL Score Prediction API

```python
# load_balancer_config.py
from flask import Flask, request, jsonify
import requests
import random
from collections import deque
import time
import threading
from datetime import datetime

app = Flask(__name__)

# ML Model Servers (like multiple ticket counters)
MODEL_SERVERS = [
    {"url": "http://10.0.1.1:5000", "healthy": True, "response_times": deque(maxlen=100)},
    {"url": "http://10.0.1.2:5000", "healthy": True, "response_times": deque(maxlen=100)},
    {"url": "http://10.0.1.3:5000", "healthy": True, "response_times": deque(maxlen=100)},
]

# Load balancing algorithms
class LoadBalancer:
    def __init__(self):
        self.current = 0
        self.request_count = {}
        
    def round_robin(self):
        """Like taking turns - Server 1, 2, 3, 1, 2, 3..."""
        healthy_servers = [s for s in MODEL_SERVERS if s["healthy"]]
        if not healthy_servers:
            return None
            
        server = healthy_servers[self.current % len(healthy_servers)]
        self.current += 1
        return server
    
    def least_connections(self):
        """Send to server with fewest active connections"""
        healthy_servers = [s for s in MODEL_SERVERS if s["healthy"]]
        if not healthy_servers:
            return None
            
        return min(healthy_servers, 
                  key=lambda s: self.request_count.get(s["url"], 0))
    
    def weighted_response_time(self):
        """Send to fastest responding server"""
        healthy_servers = [s for s in MODEL_SERVERS if s["healthy"]]
        if not healthy_servers:
            return None
            
        # Choose server with lowest average response time
        best_server = min(healthy_servers, 
                         key=lambda s: sum(s["response_times"]) / max(len(s["response_times"]), 1))
        return best_server

lb = LoadBalancer()

# Health check (like checking if ticket counter is open)
def health_check():
    """Check each server every 10 seconds"""
    while True:
        for server in MODEL_SERVERS:
            try:
                response = requests.get(f"{server['url']}/health", timeout=2)
                server["healthy"] = response.status_code == 200
            except:
                server["healthy"] = False
                print(f"❌ Server {server['url']} is down!")
        
        time.sleep(10)

# Start health checker in background
health_thread = threading.Thread(target=health_check, daemon=True)
health_thread.start()

# Main load balancer endpoint
@app.route('/predict/ipl-score', methods=['POST'])
def predict_ipl_score():
    """
    Distribute IPL score prediction requests
    During IPL season: 1 lakh+ requests per minute!
    """
    start_time = time.time()
    
    # Choose server using load balancing algorithm
    server = lb.round_robin()  # or lb.least_connections() or lb.weighted_response_time()
    
    if not server:
        # All servers down! Disaster recovery
        return jsonify({
            "error": "Service temporarily unavailable",
            "message": "Please try again in few seconds",
            "status": "503"
        }), 503
    
    try:
        # Forward request to chosen server
        lb.request_count[server["url"]] = lb.request_count.get(server["url"], 0) + 1
        
        response = requests.post(
            f"{server['url']}/predict",
            json=request.json,
            timeout=5
        )
        
        # Record response time for intelligent routing
        response_time = time.time() - start_time
        server["response_times"].append(response_time)
        
        # Add server info to response (for debugging)
        result = response.json()
        result["_server"] = server["url"]
        result["_response_time"] = f"{response_time*1000:.0f}ms"
        
        lb.request_count[server["url"]] -= 1
        return jsonify(result)
        
    except requests.exceptions.Timeout:
        # Server too slow, mark as unhealthy
        server["healthy"] = False
        return predict_ipl_score()  # Retry with different server
        
    except Exception as e:
        lb.request_count[server["url"]] -= 1
        return jsonify({"error": str(e)}), 500

# Monitoring endpoint
@app.route('/status')
def status():
    """Check load balancer status"""
    total_requests = sum(lb.request_count.values())
    healthy_count = sum(1 for s in MODEL_SERVERS if s["healthy"])
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "healthy_servers": f"{healthy_count}/{len(MODEL_SERVERS)}",
        "active_requests": total_requests,
        "uptime": "99.98%",  # Calculate from logs
        "servers": [
            {
                "url": s["url"],
                "status": "🟢 Healthy" if s["healthy"] else "🔴 Down",
                "avg_response_time": f"{sum(s['response_times'])/max(len(s['response_times']), 1)*1000:.0f}ms",
                "active_requests": lb.request_count.get(s["url"], 0)
            }
            for s in MODEL_SERVERS
        ]
    })

# Auto-scaling logic
def auto_scale():
    """
    Add more servers during high load
    Like opening more counters during Tatkal booking!
    """
    while True:
        total_requests = sum(lb.request_count.values())
        
        if total_requests > 1000 and len(MODEL_SERVERS) < 10:
            # High load - add server
            new_server = spawn_new_ml_server()
            MODEL_SERVERS.append(new_server)
            print(f"🚀 Scaled up! Added server: {new_server['url']}")
            
        elif total_requests < 100 and len(MODEL_SERVERS) > 3:
            # Low load - remove server
            server_to_remove = MODEL_SERVERS.pop()
            terminate_ml_server(server_to_remove)
            print(f"💤 Scaled down! Removed server: {server_to_remove['url']}")
            
        time.sleep(30)

# Circuit breaker pattern
class CircuitBreaker:
    """
    Like an electrical circuit breaker - prevents cascade failures
    If server fails too much, stop sending requests temporarily
    """
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_counts = {}
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.circuit_open_time = {}
    
    def call_with_circuit_breaker(self, server_url, request_func):
        # Check if circuit is open
        if server_url in self.circuit_open_time:
            if time.time() - self.circuit_open_time[server_url] < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                # Try to close circuit
                del self.circuit_open_time[server_url]
                self.failure_counts[server_url] = 0
        
        try:
            result = request_func()
            self.failure_counts[server_url] = 0  # Reset on success
            return result
        except:
            self.failure_counts[server_url] = self.failure_counts.get(server_url, 0) + 1
            
            if self.failure_counts[server_url] >= self.failure_threshold:
                self.circuit_open_time[server_url] = time.time()
                print(f"⚡ Circuit breaker OPENED for {server_url}")
            
            raise

# Example client code
"""
# During IPL final - millions of requests!
import concurrent.futures
import requests

def get_score_prediction(match_data):
    response = requests.post(
        "http://loadbalancer.ipl.com/predict/ipl-score",
        json=match_data
    )
    return response.json()

# Simulate high load
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    match_data = {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings", 
        "venue": "Wankhede",
        "current_score": 150,
        "wickets": 3,
        "overs": 15
    }
    
    # Send 10000 requests
    futures = [executor.submit(get_score_prediction, match_data) 
              for _ in range(10000)]
    
    results = [f.result() for f in futures]
    print(f"Successfully processed {len(results)} predictions!")
"""

# Run with: python load_balancer_config.py
```

This load balancer ensures:
- **No single point of failure**: If one server dies, others take over
- **Better performance**: Requests distributed evenly
- **Auto-scaling**: More servers during IPL finals, fewer during off-season
- **99.9%+ uptime**: Your service stays up even during peak load! 

---

## 6. ECR (Elastic Container Registry) {#ecr}

### What is ECR?

ECR is like a secure locker room where you store your Docker containers. It's Amazon's private Docker Hub.

**Simple Analogy**: 
- **Docker Image**: Like a tiffin box with your ML model packed inside
- **ECR**: Like a refrigerator where you store multiple tiffin boxes
- **Access Control**: Only family members (your team) can open the fridge

### Why We Need ECR?

**Real Scenario**: Your team builds fraud detection models for 5 different banks:
- **Without ECR**: Email Docker images (20GB each) to everyone 📧😱
- **With ECR**: Push once, everyone pulls when needed 🚀

**Benefits**:
1. **Security**: Your SBI fraud model stays private
2. **Version Control**: Keep old models (like old family recipes)
3. **Fast Distribution**: Teams in Mumbai, Bangalore, Delhi get same image
4. **Integration**: Works seamlessly with AWS services

### Where It Stands in the Pipeline?

ECR is the **STORAGE** for your containerized models:
```
Build Docker Image → Push to ECR → Pull from ECR → Deploy to Production
                           ↓
                   Tagged Versions:
                   - fraud-model:v1.0
                   - fraud-model:v1.1-diwali-special
                   - fraud-model:latest
```

### Real-World Alternatives?

1. **Docker Hub**: Public (like GitHub for containers)
2. **Google Container Registry**: GCP's version
3. **Azure Container Registry**: Microsoft's version
4. **Self-hosted Registry**: Your own server (more work)
5. **JFrog Artifactory**: Enterprise solution

### Code Example: Storing Indian Bank's Credit Risk Model

```bash
# Step 1: Create Dockerfile for your model
cat > Dockerfile << 'EOF'
# Start with Python image (like base ingredients for dosa)
FROM python:3.9-slim

# Install dependencies (like gathering spices)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (like adding the batter)
COPY credit_risk_model.pkl .
COPY app.py .
COPY config/ config/

# Expose port (like opening the shop)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
EOF
```

```python
# ecr_operations.py
import boto3
import docker
import base64
from datetime import datetime

class ECRManager:
    """Manage Docker images for Indian financial models"""
    
    def __init__(self, region='ap-south-1'):  # Mumbai region
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.docker_client = docker.from_env()
        self.registry_url = None
        
    def create_repository(self, bank_name, model_type):
        """
        Create ECR repository for each bank's models
        Like creating separate lockers for SBI, HDFC, ICICI
        """
        repo_name = f"{bank_name.lower()}/{model_type}"
        
        try:
            response = self.ecr_client.create_repository(
                repositoryName=repo_name,
                tags=[
                    {'Key': 'Bank', 'Value': bank_name},
                    {'Key': 'Model', 'Value': model_type},
                    {'Key': 'Compliance', 'Value': 'RBI-Approved'}
                ],
                imageScanningConfiguration={'scanOnPush': True},  # Security scan
                encryptionConfiguration={'encryptionType': 'AES256'}  # Encryption
            )
            
            self.registry_url = response['repository']['repositoryUri']
            print(f"✅ Created repository: {self.registry_url}")
            return self.registry_url
            
        except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
            print(f"Repository {repo_name} already exists")
            return self.get_repository_uri(repo_name)
    
    def docker_login(self):
        """Login to ECR (like entering PIN for locker)"""
        token = self.ecr_client.get_authorization_token()
        username = 'AWS'
        password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')[1]
        registry = token['authorizationData'][0]['proxyEndpoint']
        
        self.docker_client.login(username=username, password=password, registry=registry)
        print("✅ Logged into ECR successfully")
    
    def build_and_push_model(self, bank_name, model_type, version):
        """
        Build and push model image to ECR
        Like packaging and storing festival sweets
        """
        # Build image
        image_tag = f"{bank_name.lower()}/{model_type}:{version}"
        
        print(f"🔨 Building Docker image: {image_tag}")
        image = self.docker_client.images.build(
            path=".",
            tag=image_tag,
            labels={
                "bank": bank_name,
                "model": model_type,
                "version": version,
                "build_date": datetime.now().isoformat(),
                "rbi_compliance": "verified"
            }
        )
        
        # Tag for ECR
        ecr_tag = f"{self.registry_url}:{version}"
        image[0].tag(ecr_tag)
        
        # Push to ECR
        print(f"📤 Pushing to ECR: {ecr_tag}")
        push_logs = self.docker_client.images.push(ecr_tag, stream=True, decode=True)
        
        for log in push_logs:
            if 'status' in log:
                print(f"  {log['status']}")
        
        print(f"✅ Successfully pushed {ecr_tag}")
        return ecr_tag
    
    def list_model_versions(self, bank_name, model_type):
        """List all versions of a model (like checking recipe versions)"""
        repo_name = f"{bank_name.lower()}/{model_type}"
        
        response = self.ecr_client.list_images(
            repositoryName=repo_name,
            filter={'tagStatus': 'TAGGED'}
        )
        
        versions = []
        for image in response['imageIds']:
            if 'imageTag' in image:
                versions.append({
                    'tag': image['imageTag'],
                    'digest': image['imageDigest'][:12],
                    'size_mb': self.get_image_size(repo_name, image['imageDigest'])
                })
        
        return sorted(versions, key=lambda x: x['tag'], reverse=True)
    
    def setup_lifecycle_policy(self, bank_name, model_type):
        """
        Auto-delete old images (like cleaning old files)
        Keep only last 10 versions to save money
        """
        repo_name = f"{bank_name.lower()}/{model_type}"
        
        lifecycle_policy = {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": "Keep only last 10 images",
                    "selection": {
                        "tagStatus": "tagged",
                        "tagPrefixList": ["v"],
                        "countType": "imageCountMoreThan",
                        "countNumber": 10
                    },
                    "action": {"type": "expire"}
                },
                {
                    "rulePriority": 2,
                    "description": "Remove untagged images after 1 day",
                    "selection": {
                        "tagStatus": "untagged",
                        "countType": "sinceImagePushed",
                        "countUnit": "days",
                        "countNumber": 1
                    },
                    "action": {"type": "expire"}
                }
            ]
        }
        
        self.ecr_client.put_lifecycle_policy(
            repositoryName=repo_name,
            lifecyclePolicyText=json.dumps(lifecycle_policy)
        )
        print(f"✅ Lifecycle policy set for {repo_name}")

# Usage example
if __name__ == "__main__":
    ecr = ECRManager()
    
    # Setup for State Bank of India
    ecr.create_repository("SBI", "credit-risk-model")
    ecr.docker_login()
    
    # Build and push model
    ecr.build_and_push_model(
        bank_name="SBI",
        model_type="credit-risk-model",
        version="v2.3-post-demonetization"
    )
    
    # List all versions
    versions = ecr.list_model_versions("SBI", "credit-risk-model")
    print("\n📋 Available versions:")
    for v in versions:
        print(f"  - {v['tag']} ({v['size_mb']}MB) [{v['digest']}]")
    
    # Setup auto-cleanup
    ecr.setup_lifecycle_policy("SBI", "credit-risk-model")

# Pulling images from ECR
"""
# On production server (like opening tiffin box at lunch)
docker login -u AWS -p $(aws ecr get-login-password) 123456789.dkr.ecr.ap-south-1.amazonaws.com
docker pull 123456789.dkr.ecr.ap-south-1.amazonaws.com/sbi/credit-risk-model:v2.3
docker run -p 8080:8080 123456789.dkr.ecr.ap-south-1.amazonaws.com/sbi/credit-risk-model:v2.3
"""
```

---

## 7. ECS (Elastic Container Service) {#ecs}

### What is ECS?

ECS is like a smart restaurant manager who decides which waiter (container) serves which table (request) and ensures all waiters are working efficiently.

**Simple Analogy**:
- **Container**: A waiter serving one dish (your ML model)
- **ECS**: Restaurant manager coordinating all waiters
- **Task**: One waiter's shift
- **Service**: Ensuring 10 waiters are always available

### Why We Need ECS?

**Real Scenario**: Paytm's fraud detection during Diwali sale:
- Normal days: 1000 transactions/second
- Diwali sale: 50,000 transactions/second

**Without ECS**: Manually start 50 servers, configure each, pray nothing crashes 😰
**With ECS**: Auto-scales from 10 to 500 containers in minutes! 🚀

### Where It Stands in the Pipeline?

ECS is the **ORCHESTRATOR** that manages your containers:
```
ECR (Storage) → ECS (Running) → Load Balancer → Users
                    ↓
            - Auto-scaling
            - Health checks  
            - Rolling updates
            - Resource management
```

### Real-World Alternatives?

1. **Kubernetes**: More complex but more features
2. **Docker Swarm**: Simpler but less powerful
3. **AWS Fargate**: Serverless version of ECS
4. **Google Cloud Run**: GCP's container service
5. **Manual Docker**: Run containers yourself (not scalable)

### Code Example: Deploying Zomato's Food Recommendation Model

```python
# ecs_deployment.py
import boto3
import json
from typing import Dict, List

class ECSDeploymentManager:
    """Deploy ML models on ECS for food delivery apps"""
    
    def __init__(self, cluster_name="zomato-ml-cluster"):
        self.ecs = boto3.client('ecs', region_name='ap-south-1')
        self.ec2 = boto3.client('ec2', region_name='ap-south-1')
        self.elbv2 = boto3.client('elbv2', region_name='ap-south-1')
        self.cluster_name = cluster_name
    
    def create_cluster(self):
        """Create ECS cluster (like opening new restaurant branch)"""
        response = self.ecs.create_cluster(
            clusterName=self.cluster_name,
            capacityProviders=['FARGATE', 'FARGATE_SPOT'],
            defaultCapacityProviderStrategy=[
                {
                    'capacityProvider': 'FARGATE',
                    'weight': 1,
                    'base': 2  # Always keep 2 containers on-demand
                },
                {
                    'capacityProvider': 'FARGATE_SPOT',
                    'weight': 4  # 80% on spot instances (save money!)
                }
            ],
            tags=[
                {'key': 'Project', 'value': 'FoodRecommendation'},
                {'key': 'Environment', 'value': 'Production'}
            ]
        )
        print(f"✅ Created cluster: {self.cluster_name}")
        return response['cluster']['clusterArn']
    
    def create_task_definition(self):
        """
        Define how to run the container (like recipe for dish)
        Task = Recipe for running one container
        """
        task_definition = {
            "family": "food-recommendation-model",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "1024",  # 1 vCPU
            "memory": "2048",  # 2 GB RAM
            "containerDefinitions": [
                {
                    "name": "recommendation-api",
                    "image": "123456789.dkr.ecr.ap-south-1.amazonaws.com/zomato/food-rec:latest",
                    "portMappings": [
                        {
                            "containerPort": 8080,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {"name": "MODEL_TYPE", "value": "collaborative_filtering"},
                        {"name": "CACHE_SIZE", "value": "1000"},
                        {"name": "MAX_RECOMMENDATIONS", "value": "20"},
                        {"name": "REGION_FILTER", "value": "INDIA"}
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": "/ecs/food-recommendation",
                            "awslogs-region": "ap-south-1",
                            "awslogs-stream-prefix": "ecs"
                        }
                    },
                    "healthCheck": {
                        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
                        "interval": 30,
                        "timeout": 5,
                        "retries": 3,
                        "startPeriod": 60
                    }
                },
                {
                    "name": "redis-cache",
                    "image": "redis:alpine",
                    "memory": 512,
                    "portMappings": [
                        {
                            "containerPort": 6379,
                            "protocol": "tcp"
                        }
                    ]
                }
            ]
        }
        
        response = self.ecs.register_task_definition(**task_definition)
        print(f"✅ Created task definition: {task_definition['family']}")
        return response['taskDefinition']['taskDefinitionArn']
    
    def create_service(self, task_definition_arn: str, target_group_arn: str):
        """
        Create ECS Service (like hiring permanent staff)
        Service = Ensures X containers are always running
        """
        service_config = {
            "cluster": self.cluster_name,
            "serviceName": "food-recommendation-service",
            "taskDefinition": task_definition_arn,
            "desiredCount": 10,  # Start with 10 containers
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": ["subnet-1", "subnet-2"],  # Multiple availability zones
                    "securityGroups": ["sg-ml-models"],
                    "assignPublicIp": "ENABLED"
                }
            },
            "loadBalancers": [
                {
                    "targetGroupArn": target_group_arn,
                    "containerName": "recommendation-api",
                    "containerPort": 8080
                }
            ],
            "healthCheckGracePeriodSeconds": 60,
            "deploymentConfiguration": {
                "maximumPercent": 200,  # Can go up to 20 containers during deployment
                "minimumHealthyPercent": 100  # Keep all containers running during deployment
            },
            "placementStrategies": [
                {
                    "type": "spread",
                    "field": "attribute:ecs.availability-zone"  # Spread across zones
                }
            ]
        }
        
        response = self.ecs.create_service(**service_config)
        print(f"✅ Created service: {service_config['serviceName']}")
        return response['service']['serviceArn']
    
    def setup_auto_scaling(self, service_name: str):
        """
        Configure auto-scaling (like calling extra waiters during rush hour)
        Scale based on CPU, memory, or custom metrics
        """
        autoscaling = boto3.client('application-autoscaling')
        
        # Register scalable target
        autoscaling.register_scalable_target(
            ServiceNamespace='ecs',
            ResourceId=f'service/{self.cluster_name}/{service_name}',
            ScalableDimension='ecs:service:DesiredCount',
            MinCapacity=5,   # Minimum 5 containers (off-peak)
            MaxCapacity=100  # Maximum 100 containers (peak hours)
        )
        
        # Scale based on CPU utilization
        autoscaling.put_scaling_policy(
            PolicyName='cpu-scaling-policy',
            ServiceNamespace='ecs',
            ResourceId=f'service/{self.cluster_name}/{service_name}',
            ScalableDimension='ecs:service:DesiredCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 70.0,  # Target 70% CPU utilization
                'PredefinedMetricType': 'ECSServiceAverageCPUUtilization',
                'ScaleInCooldown': 300,   # Wait 5 min before scaling down
                'ScaleOutCooldown': 60    # Wait 1 min before scaling up
            }
        )
        
        # Scale based on time (lunch/dinner rush)
        autoscaling.put_scheduled_action(
            ServiceNamespace='ecs',
            ResourceId=f'service/{self.cluster_name}/{service_name}',
            ScalableDimension='ecs:service:DesiredCount',
            ScheduledActionName='lunch-rush',
            Schedule='cron(30 11 * * ? *)',  # 12:30 PM IST daily
            MinCapacity=50,
            MaxCapacity=100
        )
        
        autoscaling.put_scheduled_action(
            ServiceNamespace='ecs',
            ResourceId=f'service/{self.cluster_name}/{service_name}',
            ScalableDimension='ecs:service:DesiredCount',
            ScheduledActionName='dinner-rush',
            Schedule='cron(30 19 * * ? *)',  # 8:30 PM IST daily
            MinCapacity=50,
            MaxCapacity=100
        )
        
        print("✅ Auto-scaling configured for lunch and dinner rush hours!")
    
    def deploy_new_version(self, service_name: str, new_task_definition: str):
        """
        Rolling deployment (like gradually changing menu items)
        No downtime - customers keep getting recommendations!
        """
        response = self.ecs.update_service(
            cluster=self.cluster_name,
            service=service_name,
            taskDefinition=new_task_definition,
            forceNewDeployment=True,
            deploymentConfiguration={
                'deploymentCircuitBreaker': {
                    'enable': True,
                    'rollback': True  # Auto-rollback if deployment fails
                },
                'maximumPercent': 200,
                'minimumHealthyPercent': 100
            }
        )
        
        print(f"🚀 Deploying new version...")
        print(f"   Old version containers will be replaced gradually")
        print(f"   No downtime expected!")
        
        # Monitor deployment
        self.monitor_deployment(service_name)
    
    def monitor_deployment(self, service_name: str):
        """Watch deployment progress (like kitchen supervisor)"""
        import time
        
        while True:
            response = self.ecs.describe_services(
                cluster=self.cluster_name,
                services=[service_name]
            )
            
            service = response['services'][0]
            deployments = service['deployments']
            
            print(f"\n📊 Deployment Status:")
            for d in deployments:
                print(f"   - {d['status']}: {d['runningCount']}/{d['desiredCount']} containers")
            
            if len(deployments) == 1 and deployments[0]['status'] == 'PRIMARY':
                print("✅ Deployment completed successfully!")
                break
                
            time.sleep(10)

# Usage Example
if __name__ == "__main__":
    ecs = ECSDeploymentManager()
    
    # Setup infrastructure
    cluster_arn = ecs.create_cluster()
    task_def_arn = ecs.create_task_definition()
    
    # Create load balancer target group (simplified)
    target_group_arn = "arn:aws:elasticloadbalancing:..."
    
    # Deploy service
    service_arn = ecs.create_service(task_def_arn, target_group_arn)
    
    # Setup auto-scaling for rush hours
    ecs.setup_auto_scaling("food-recommendation-service")
    
    # Deploy new version (e.g., with Diwali special recommendations)
    new_task_def = "food-recommendation-model:diwali-special"
    ecs.deploy_new_version("food-recommendation-service", new_task_def)

# Container logs example
"""
# View logs (like checking kitchen activity)
aws logs tail /ecs/food-recommendation --follow

# Sample logs:
2024-01-15 12:30:45 INFO: Recommendation request from Mumbai
2024-01-15 12:30:45 INFO: User preferences: North Indian, Vegetarian
2024-01-15 12:30:46 INFO: Generated 20 recommendations in 187ms
2024-01-15 12:30:46 INFO: Top recommendation: Paneer Butter Masala
"""
```

---

## 8. Kubernetes (K8s) {#kubernetes}

### What is Kubernetes?

Kubernetes is like a super-smart city traffic controller that manages thousands of vehicles (containers) across multiple roads (servers), ensuring everyone reaches their destination efficiently.

**Simple Analogy**:
- **Pod**: An auto-rickshaw (can carry 1-3 containers)
- **Node**: A road/street (physical server)
- **Cluster**: The entire city road network
- **Service**: Traffic signals directing requests
- **Deployment**: Fleet management (ensuring 100 autos are always running)

### Why We Need Kubernetes?

**Real Scenario**: Flipkart's Big Billion Day Sale
- **Normal Day**: 1,000 ML models serving predictions
- **Sale Day**: Need 50,000 models across 10 data centers
- **Challenge**: Coordinate across Mumbai, Bangalore, Delhi centers

**Without Kubernetes**: 😱 Chaos! Manual management impossible
**With Kubernetes**: 😎 Automatic orchestration across all centers

### Where It Stands in the Pipeline?

Kubernetes is the **MASTER ORCHESTRATOR**:
```
Code → Docker Image → Container Registry → [KUBERNETES] → Production
                                                ↓
                                    - Manages 1000s of containers
                                    - Auto-healing
                                    - Load distribution  
                                    - Multi-region deployment
```

### Real-World Alternatives?

1. **ECS**: Simpler but AWS-only
2. **Docker Swarm**: Easier but less powerful
3. **OpenShift**: Enterprise Kubernetes
4. **Nomad**: HashiCorp's alternative
5. **Manual Management**: Not realistic for scale

### Code Example: Deploying IRCTC's Ticket Price Prediction System

```yaml
# irctc-ml-deployment.yaml
# Deploy ML models for predicting ticket prices and availability

---
# Namespace (like creating separate department)
apiVersion: v1
kind: Namespace
metadata:
  name: irctc-ml
  labels:
    project: "railway-predictions"
    
---
# ConfigMap (like settings file)
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: irctc-ml
data:
  MODEL_TYPE: "dynamic_pricing"
  PEAK_HOURS: "6-9,17-20"  # Morning and evening rush
  TATKAL_WINDOW: "10:00"
  MAX_PREDICTIONS_PER_SECOND: "10000"
  CACHE_DURATION: "300"
  
---
# Secret (like keeping passwords in safe)
apiVersion: v1
kind: Secret
metadata:
  name: model-secrets
  namespace: irctc-ml
type: Opaque
data:
  DATABASE_URL: "cG9zdGdyZXNxbDovL3VzZXI6cGFzc0BkYi5pcmN0Yy5jb20vdGlja2V0cw=="
  API_KEY: "YWJjZGVmZ2hpams="
  
---
# Deployment (like hiring staff)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: price-prediction-model
  namespace: irctc-ml
  labels:
    app: price-predictor
spec:
  replicas: 50  # Start with 50 instances
  selector:
    matchLabels:
      app: price-predictor
  template:
    metadata:
      labels:
        app: price-predictor
        version: "v2.5"
    spec:
      containers:
      - name: prediction-api
        image: irctc.azurecr.io/price-predictor:v2.5-republic-day
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        
        # Resource requirements (like seat allocation)
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"  # 1 CPU
          limits:
            memory: "4Gi"
            cpu: "2000m"  # 2 CPUs
        
        # Environment variables
        envFrom:
        - configMapRef:
            name: model-config
        - secretRef:
            name: model-secrets
        
        # Health checks (like ticket checker)
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        
        # Volume mounts (like luggage compartment)
        volumeMounts:
        - name: model-cache
          mountPath: /cache
        - name: ml-models
          mountPath: /models
          
      # Init container (like cleaning train before journey)
      initContainers:
      - name: model-downloader
        image: irctc.azurecr.io/model-downloader:latest
        command: ['sh', '-c', 'download-models.sh']
        volumeMounts:
        - name: ml-models
          mountPath: /models
          
      # Volumes (like train compartments)
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 10Gi
      - name: ml-models
        persistentVolumeClaim:
          claimName: model-storage
          
---
# Service (like booking counter)
apiVersion: v1
kind: Service
metadata:
  name: price-prediction-service
  namespace: irctc-ml
spec:
  selector:
    app: price-predictor
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
  
---
# HorizontalPodAutoscaler (like adding more trains during festivals)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: price-predictor-hpa
  namespace: irctc-ml
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: price-prediction-model
  minReplicas: 50
  maxReplicas: 500  # Can scale up to 500 pods!
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Custom metric - scale based on request rate
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"  # 1000 requests/second per pod
        
  # Scale up/down behavior
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 10  # Remove maximum 10% pods at a time
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30  # Scale up quickly
      policies:
      - type: Percent
        value: 100  # Can double the pods
        periodSeconds: 30
      - type: Pods
        value: 20  # Or add 20 pods
        periodSeconds: 30
      selectPolicy: Max  # Choose whichever adds more pods
```

```python
# kubernetes_operations.py
from kubernetes import client, config
import yaml
import time
from datetime import datetime

class KubernetesMLOps:
    """Manage ML deployments on Kubernetes"""
    
    def __init__(self):
        # Load kubeconfig
        config.load_incluster_config()  # If running inside cluster
        # OR
        # config.load_kube_config()  # If running outside cluster
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v2 = client.AutoscalingV2Api()
        
    def deploy_model(self, model_name: str, image: str, replicas: int = 10):
        """Deploy ML model to Kubernetes"""
        
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=f"{model_name}-deployment"),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": model_name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(labels={"app": model_name}),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=model_name,
                                image=image,
                                ports=[client.V1ContainerPort(container_port=8080)],
                                resources=client.V1ResourceRequirements(
                                    requests={"cpu": "1", "memory": "2Gi"},
                                    limits={"cpu": "2", "memory": "4Gi"}
                                ),
                                env=[
                                    client.V1EnvVar(name="MODEL_NAME", value=model_name),
                                    client.V1EnvVar(name="LOG_LEVEL", value="INFO")
                                ]
                            )
                        ]
                    )
                )
            )
        )
        
        # Create deployment
        self.apps_v1.create_namespaced_deployment(
            namespace="default",
            body=deployment
        )
        print(f"✅ Deployed {model_name} with {replicas} replicas")
        
    def rolling_update(self, deployment_name: str, new_image: str):
        """
        Update model with zero downtime
        Like changing train engine while train is running!
        """
        # Get current deployment
        deployment = self.apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace="default"
        )
        
        # Update image
        deployment.spec.template.spec.containers[0].image = new_image
        
        # Apply update
        self.apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace="default",
            body=deployment
        )
        
        print(f"🚀 Rolling update started for {deployment_name}")
        self.monitor_rollout(deployment_name)
        
    def monitor_rollout(self, deployment_name: str):
        """Monitor deployment progress"""
        while True:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace="default"
            )
            
            conditions = deployment.status.conditions
            for condition in conditions:
                if condition.type == "Progressing":
                    print(f"📊 Status: {condition.message}")
                    
            if deployment.status.ready_replicas == deployment.spec.replicas:
                print("✅ Rollout completed successfully!")
                break
                
            time.sleep(5)
            
    def scale_for_peak_hours(self):
        """
        Scale up during Tatkal booking time
        Like adding more booking counters at 10 AM!
        """
        tatkal_deployments = [
            "ticket-availability-predictor",
            "price-calculator",
            "seat-recommender",
            "payment-processor"
        ]
        
        current_hour = datetime.now().hour
        
        if current_hour == 10:  # Tatkal time!
            print("⏰ Tatkal time detected! Scaling up...")
            
            for deployment in tatkal_deployments:
                try:
                    # Scale to 200 replicas
                    self.apps_v1.patch_namespaced_deployment_scale(
                        name=deployment,
                        namespace="default",
                        body={"spec": {"replicas": 200}}
                    )
                    print(f"  ✅ Scaled {deployment} to 200 replicas")
                except Exception as e:
                    print(f"  ❌ Failed to scale {deployment}: {e}")
                    
        elif current_hour == 12:  # Post-Tatkal
            print("😌 Tatkal rush over. Scaling down...")
            
            for deployment in tatkal_deployments:
                try:
                    # Scale back to 20 replicas
                    self.apps_v1.patch_namespaced_deployment_scale(
                        name=deployment,
                        namespace="default",
                        body={"spec": {"replicas": 20}}
                    )
                    print(f"  ✅ Scaled {deployment} back to 20 replicas")
                except Exception as e:
                    print(f"  ❌ Failed to scale {deployment}: {e}")
    
    def canary_deployment(self, model_name: str, new_version: str, canary_percentage: int = 10):
        """
        Gradual rollout (like testing new train on few routes first)
        10% traffic to new version, 90% to old version
        """
        # This is simplified - in production use Istio or Flagger
        print(f"🐤 Starting canary deployment for {model_name}")
        print(f"   {canary_percentage}% → {new_version}")
        print(f"   {100-canary_percentage}% → current version")
        
        # Deploy canary version with fewer replicas
        total_replicas = 100
        canary_replicas = int(total_replicas * canary_percentage / 100)
        stable_replicas = total_replicas - canary_replicas
        
        # Update stable version replica count
        self.apps_v1.patch_namespaced_deployment_scale(
            name=f"{model_name}-stable",
            namespace="default",
            body={"spec": {"replicas": stable_replicas}}
        )
        
        # Deploy canary version
        self.deploy_model(
            model_name=f"{model_name}-canary",
            image=new_version,
            replicas=canary_replicas
        )
        
        print(f"✅ Canary deployment active:")
        print(f"   Stable: {stable_replicas} pods")
        print(f"   Canary: {canary_replicas} pods")
        
    def get_pod_metrics(self, namespace: str = "irctc-ml"):
        """Get resource usage (like checking train occupancy)"""
        pods = self.v1.list_namespaced_pod(namespace=namespace)
        
        print(f"\n📊 Pod Status in {namespace}:")
        print(f"{'Pod Name':<40} {'Status':<15} {'CPU':<10} {'Memory':<10}")
        print("-" * 80)
        
        for pod in pods.items:
            status = pod.status.phase
            
            # Get metrics (simplified - use metrics-server in production)
            cpu = "N/A"
            memory = "N/A"
            
            print(f"{pod.metadata.name:<40} {status:<15} {cpu:<10} {memory:<10}")

# Usage example
if __name__ == "__main__":
    k8s = KubernetesMLOps()
    
    # Deploy fraud detection model
    k8s.deploy_model(
        model_name="upi-fraud-detector",
        image="gcr.io/irctc-ml/fraud-detector:v3.2",
        replicas=50
    )
    
    # Scale for Tatkal time
    k8s.scale_for_peak_hours()
    
    # Canary deployment for new model
    k8s.canary_deployment(
        model_name="ticket-price-predictor",
        new_version="gcr.io/irctc-ml/price-predictor:v4.0-festival",
        canary_percentage=10
    )
    
    # Monitor pods
    k8s.get_pod_metrics()

# Kubectl commands (like train control commands)
"""
# View all ML deployments
kubectl get deployments -n irctc-ml

# Check pod status
kubectl get pods -n irctc-ml

# View logs (like CCTV footage)
kubectl logs -f deployment/price-prediction-model -n irctc-ml

# Scale manually
kubectl scale deployment price-prediction-model --replicas=100 -n irctc-ml

# Check resource usage
kubectl top pods -n irctc-ml

# Emergency rollback
kubectl rollout undo deployment/price-prediction-model -n irctc-ml
"""
```

Kubernetes provides:
- **Massive Scale**: Handle millions of requests across data centers
- **Self-Healing**: Automatically replaces failed containers
- **Rolling Updates**: Zero-downtime deployments
- **Resource Optimization**: Efficiently uses available hardware
- **Multi-Cloud**: Works on AWS, Azure, GCP, or on-premise! 

---

## 9. Fargate {#fargate}

### What is Fargate?

Fargate is like hiring a taxi instead of buying a car - you don't worry about maintenance, parking, or fuel. AWS manages everything; you just say where you want to go!

**Simple Analogy**:
- **EC2**: Buying your own car (manage everything)
- **ECS**: Hiring a driver for your car (manage less)
- **Fargate**: Taking Uber/Ola (manage nothing!)

### Why We Need Fargate?

**Real Scenario**: Startup building UPI fraud detection
- **Without Fargate**: Buy servers, install OS, configure security, manage updates 😰
- **With Fargate**: Just give your container, AWS handles everything! 🎉

**Perfect for**:
- Startups (no DevOps team needed)
- Unpredictable workloads (pay per use)
- Quick experiments (launch in minutes)

### Where It Stands in the Pipeline?

Fargate is the **SERVERLESS CONTAINER** platform:
```
Docker Image → Fargate → Running Container
                ↓
        No servers to manage!
        Auto-scaling built-in
        Pay only when running
```

### Real-World Alternatives?

1. **EC2**: Full control but more work
2. **Lambda**: For smaller, event-driven tasks
3. **Google Cloud Run**: GCP's serverless containers
4. **Azure Container Instances**: Microsoft's version
5. **Kubernetes + Virtual Kubelet**: Complex setup

### Code Example: Deploy Paytm Transaction Analyzer

```bash
# Task definition (2 lines)
aws ecs register-task-definition --family paytm-analyzer --cpu 512 --memory 1024 \
  --requires-compatibilities FARGATE --network-mode awsvpc

# Run task (1 line)
aws ecs run-task --cluster prod --task-definition paytm-analyzer --launch-type FARGATE
```

That's it! No EC2 instances, no server management - container runs immediately!

---

## 10. EC2 (Elastic Compute Cloud) {#ec2}

### What is EC2?

EC2 is like renting a computer in AWS cloud - you get full control as if it's your own laptop, but it lives in Amazon's data center.

**Simple Analogy**:
- **Your Laptop**: Physical computer at home
- **EC2**: Virtual computer in cloud
- **Benefit**: Can get 1000 laptops in 2 minutes!

### Why We Need EC2?

**Real Scenario**: BigBasket's Diwali Sale
- Need 100 servers for 1 week only
- **Without EC2**: Buy 100 computers for ₹50 lakhs, use for 1 week 😱
- **With EC2**: Rent 100 servers for 1 week, pay ₹50,000 😊

**Use Cases**:
- Training large ML models (rent GPU servers)
- Hosting websites/APIs
- Development environments
- Big data processing

### Where It Stands in the Pipeline?

EC2 is the **FOUNDATION** layer:
```
EC2 Instance → Install Docker → Run Containers → Deploy ML Models
      ↓
  Full control
  Any software
  Any configuration
```

### Real-World Alternatives?

1. **Physical Servers**: Buy and manage yourself
2. **Google Compute Engine**: GCP's version
3. **Azure Virtual Machines**: Microsoft's version
4. **DigitalOcean Droplets**: Simpler, cheaper
5. **Bare Metal Servers**: Even more control

### Code Example: Launch ML Training Server

```bash
# Launch GPU instance for model training (2 lines)
aws ec2 run-instances --image-id ami-0abcdef1234567890 \
  --instance-type p3.2xlarge --key-name my-key --count 1

# Connect to instance (1 line)
ssh -i my-key.pem ec2-user@<instance-ip>
```

---

## 11. Deployment {#deployment}

### What is Deployment?

Deployment is moving your ML model from laptop to production - like opening a new restaurant branch after perfecting recipes at home.

**Simple Analogy**:
- **Development**: Cooking at home
- **Testing**: Friends taste your food
- **Deployment**: Opening restaurant for public
- **Production**: Serving 1000s of customers daily

### Why We Need Proper Deployment?

**Real Scenario**: Swiggy's Restaurant Recommendation Model
- Data Scientist: "Model works perfectly on my laptop!"
- Production: Crashes with 1000 users 💥

**Proper Deployment Ensures**:
- Works at scale (millions of users)
- High availability (99.9% uptime)
- Fast response (<100ms)
- Easy updates (no downtime)

### Where It Stands in the Pipeline?

Deployment is the **BRIDGE TO PRODUCTION**:
```
Train Model → Test Model → [DEPLOYMENT] → Production
                                ↓
                        - Package model
                        - Create API
                        - Configure infrastructure
                        - Monitor performance
```

### Real-World Alternatives?

1. **Manual Copy**: Copy files to server (risky!)
2. **FTP Upload**: Old school method
3. **GitHub Pages**: For static sites only
4. **Heroku**: Simple but limited
5. **Netlify/Vercel**: For web apps

### Code Example: Deploy Model with Docker

```python
# Create API (app.py) - 3 lines
from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict(): return jsonify({"prediction": model.predict(request.json)})
```

```bash
# Deploy (2 lines)
docker build -t fraud-model . && docker push myregistry/fraud-model:v1
kubectl apply -f deployment.yaml
```

---

## 12. CI/CD (Continuous Integration/Continuous Deployment) {#cicd}

### What is CI/CD?

CI/CD is like a conveyor belt in a dosa factory - automatically takes batter, makes dosas, packs them, and delivers to stores without manual intervention.

**Simple Analogy**:
- **CI (Continuous Integration)**: Auto-checking if dosa batter is good
- **CD (Continuous Deployment)**: Auto-delivering fresh dosas to all outlets

### Why We Need CI/CD?

**Real Scenario**: PhonePe Payment Processing
- 50 developers updating code daily
- **Without CI/CD**: Manual testing, manual deployment = 2 days per release 😴
- **With CI/CD**: Auto test, auto deploy = 20 minutes per release! 🚀

**Benefits**:
- Catch bugs early (before customer sees)
- Deploy faster (multiple times per day)
- Less human error (automation)
- Happy developers (less manual work)

### Where It Stands in the Pipeline?

CI/CD is the **AUTOMATION HIGHWAY**:
```
Code Push → [CI/CD Pipeline] → Production
               ↓
         - Run tests
         - Build images
         - Deploy to staging
         - Deploy to production
```

### Real-World Alternatives?

1. **Manual Process**: Test and deploy by hand
2. **Shell Scripts**: Basic automation
3. **Cron Jobs**: Schedule deployments
4. **GitHub Actions**: Simple CI/CD
5. **GitLab CI**: Built into GitLab

### Code Example: Simple CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml (3 lines)
on: push
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - run: docker build -t myapp . && docker push myapp:latest
```

---

## 13. Jenkins {#jenkins}

### What is Jenkins?

Jenkins is like a smart assistant who watches your code, runs tests, and deploys automatically - like having a 24/7 employee who never takes breaks!

**Simple Analogy**:
- **Jenkins**: Your automated assistant
- **Jobs**: Tasks you give to assistant
- **Pipeline**: Step-by-step instructions
- **Triggers**: "Do this when that happens"

### Why We Need Jenkins?

**Real Scenario**: Flipkart's Search Algorithm
- Team updates ML model daily
- Need to test with 1000s of queries
- **Without Jenkins**: 4 hours manual testing daily 😓
- **With Jenkins**: Automatic testing in 30 minutes! ✨

**Jenkins Automates**:
- Running tests (does model work?)
- Building packages (create deployable version)
- Deployment (push to production)
- Notifications (tell team if something breaks)

### Where It Stands in the Pipeline?

Jenkins is the **AUTOMATION BUTLER**:
```
Developer pushes code → [Jenkins] → Tests → Build → Deploy
                           ↓
                   - Watches GitHub
                   - Runs on schedule
                   - Triggers on events
```

### Real-World Alternatives?

1. **CircleCI**: Cloud-based, easier
2. **Travis CI**: Popular with open source
3. **GitLab CI**: If using GitLab
4. **GitHub Actions**: Native to GitHub
5. **TeamCity**: Enterprise option

### Code Example: Jenkins Pipeline for ML Model

```groovy
// Jenkinsfile (3 lines)
pipeline {
    stages {
        stage('Test') { steps { sh 'python test_model.py' }}
        stage('Deploy') { steps { sh 'kubectl apply -f model.yaml' }}
    }
}
```

---

## 14. MLflow {#mlflow}

### What is MLflow?

MLflow is like a lab notebook for ML experiments - tracks every experiment, every parameter, every result. Never lose track of which model worked best!

**Simple Analogy**:
- **Without MLflow**: Cooking without writing recipes (forget what worked)
- **With MLflow**: Detailed recipe book with every variation tried
- **Benefit**: Can recreate that perfect biryani from 6 months ago!

### Why We Need MLflow?

**Real Scenario**: Myntra's Fashion Recommendation
- Tried 100 different models
- Different parameters each time
- **Without MLflow**: "Which model gave 95% accuracy?" 🤔
- **With MLflow**: Click and see exact model, parameters, data! 📊

**MLflow Tracks**:
- Parameters (learning rate, epochs)
- Metrics (accuracy, loss)
- Models (save every version)
- Code (exact version used)

### Where It Stands in the Pipeline?

MLflow is the **EXPERIMENT TRACKER**:
```
Experiment → [MLflow] → Best Model → Production
                ↓
         - Log parameters
         - Track metrics
         - Version models
         - Compare results
```

### Real-World Alternatives?

1. **Weights & Biases**: More features, cloud-based
2. **Neptune.ai**: Good visualization
3. **Comet.ml**: Easy integration
4. **TensorBoard**: For deep learning
5. **Excel Sheets**: Please don't! 😅

### Code Example: Track IPL Win Prediction Model

```python
import mlflow
# Track experiment (3 lines)
mlflow.start_run()
mlflow.log_param("algorithm", "RandomForest")
mlflow.log_metric("accuracy", 0.89)
mlflow.sklearn.log_model(model, "ipl-predictor")
```

---

## 15. SageMaker {#sagemaker}

### What is SageMaker?

SageMaker is like a complete ML kitchen by Amazon - has everything from cutting board (data prep) to oven (training) to serving plates (deployment).

**Simple Analogy**:
- **DIY ML**: Buy ingredients, cook, serve yourself
- **SageMaker**: 5-star hotel kitchen - just tell what you want!
- **Includes**: Notebooks, training, deployment, monitoring

### Why We Need SageMaker?

**Real Scenario**: Ola's Surge Pricing Model
- Need to train on 1TB data
- Requires 50 GPUs
- Must deploy to 10 cities
- **Without SageMaker**: Setup everything manually (2 weeks) 😰
- **With SageMaker**: Click few buttons (2 hours) 🎉

**SageMaker Provides**:
- Jupyter notebooks (development)
- Distributed training (100s of GPUs)
- AutoML (automatic model selection)
- Easy deployment (one-click)
- Built-in monitoring

### Where It Stands in the Pipeline?

SageMaker is the **END-TO-END PLATFORM**:
```
SageMaker Studio → Train → Deploy → Monitor
        ↓
  Complete ML lifecycle
  All in one place
  Fully managed by AWS
```

### Real-World Alternatives?

1. **Google Vertex AI**: GCP's version
2. **Azure ML**: Microsoft's platform
3. **Databricks**: Good for Spark users
4. **Kubeflow**: Open source, complex
5. **Build Your Own**: Lots of work!

### Code Example: Train and Deploy Model

```python
# Train model on SageMaker (3 lines)
from sagemaker import RandomForest
model = RandomForest(role='arn:aws:iam::role', instance_type='ml.m5.xlarge')
model.fit('s3://bucket/train-data')
model.deploy(instance_type='ml.t2.medium')
```

---

## 16. Google Cloud Bucket (GCS) {#gcs}

### What is Google Cloud Bucket?

GCS is like a massive digital almari (cupboard) where you can store unlimited data - photos, videos, datasets, models - accessible from anywhere!

**Simple Analogy**:
- **Hard Drive**: 1TB storage at home
- **Google Drive**: 15GB personal storage
- **GCS Bucket**: Unlimited storage for business
- **Cost**: Pay only for what you use

### Why We Need Cloud Storage?

**Real Scenario**: Hotstar IPL Streaming Data
- 50TB video highlights daily
- Accessed by crores of users
- **Without Cloud**: Buy 1000 hard drives 💾
- **With GCS**: Upload and forget! ☁️

**Benefits**:
- Unlimited storage (petabytes!)
- Access from anywhere
- Automatic backups
- Pay per GB (₹1.5/GB/month)
- 99.999999999% durability

### Where It Stands in the Pipeline?

GCS is the **DATA LAKE**:
```
Raw Data → [GCS Bucket] → Process → Train Model
              ↓
        - Store datasets
        - Save models
        - Keep backups
        - Share with team
```

### Real-World Alternatives?

1. **AWS S3**: Amazon's version (most popular)
2. **Azure Blob Storage**: Microsoft's version
3. **Dropbox Business**: Simpler but expensive
4. **MinIO**: Self-hosted S3 compatible
5. **HDFS**: For Hadoop users

### Code Example: Store IPL Match Data

```python
from google.cloud import storage
# Upload data (3 lines)
client = storage.Client()
bucket = client.bucket('ipl-match-data')
blob = bucket.blob('match-2024-final.csv')
blob.upload_from_filename('local-match-data.csv')
```

```bash
# Download model (1 line)
gsutil cp gs://my-models/fraud-detector-v2.pkl ./
```

---

## Summary: When to Use What?

### For Beginners Starting MLOps:
1. **Start with**: Git (version control) + Streamlit (demos)
2. **Then add**: Docker + API basics
3. **Finally**: CI/CD + Cloud deployment

### For DevOps Professionals:
1. **Leverage**: Your Kubernetes/Docker knowledge
2. **Learn**: MLflow for experiment tracking
3. **Master**: Model serving and monitoring

### For Different Scales:
- **Startup/POC**: Streamlit + Heroku/Lambda
- **Small Team**: EC2 + Docker + Jenkins
- **Medium Company**: ECS/Kubernetes + MLflow
- **Enterprise**: SageMaker/Vertex AI + Full CI/CD

### Cost Considerations:
- **Cheapest**: EC2 Spot Instances + S3
- **Balanced**: Fargate + ECR
- **Premium**: SageMaker end-to-end
- **Best Value**: Mix and match based on needs

Remember: **Start simple, add complexity only when needed!** 🚀 