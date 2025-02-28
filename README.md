# Experiment Management System

## Overview
This project is a full-stack **Experiment Management System** that allows users to run machine learning experiments with different hyperparameters, track training progress in real time, and view experiment results. The system consists of a **FastAPI backend** for experiment processing and a **React frontend** for an interactive user interface.

---

## Features
- Define hyperparameters (learning rate, batch size, epochs) for experiments.
- Run multiple experiments concurrently.
- Display real-time training progress (current epoch, loss values).
- Store and retrieve experiment results from a database.
- Sort and compare experiments based on accuracy and loss.
- Automatically remove stuck experiments running for too long.
- Persist experiment state even after UI reload.
---

## Tech Stack
### **Backend (FastAPI + PostgreSQL)**
- **FastAPI** - API framework for managing experiments.
- **SQLAlchemy** - ORM for database interactions.
- **PostgreSQL** - Database to store experiment results.
- **Pydantic** - Data validation and schema handling.
- **Uvicorn** - ASGI server for FastAPI.
- **PyTorch** - Deep learning framework for training models.

### **Frontend (React + TailwindCSS)**
- **React.js** - UI framework for managing experiments.
- **Axios** - HTTP client for API communication.
---

## Installation & Setup

### **1. Backend Setup**
#### **Prerequisites:**
- Python 3.9+
- PostgreSQL
- Virtual Environment (`venv` or `conda`)

#### **Installation Steps:**
```sh
# Clone the repository
git clone https://github.com/your-repo/experiment-management-system.git
cd experiment-management-system/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_db.py

# Run the backend server
uvicorn api:app --reload
```

---

### **2. Frontend Setup**
#### **Prerequisites:**
- Node.js 16+
- npm or yarn

#### **Installation Steps:**
```sh
cd ../frontend

# Install dependencies
npm install

# Start the React frontend
yarn start  # or npm start
```

---

## API Endpoints
### **Experiment Management**
| Method | Endpoint               | Description                       |
|--------|------------------------|-----------------------------------|
| GET    | `/experiments/`         | Fetch all experiments            |
| POST   | `/start_experiment/`    | Start a new experiment           |
---

## Contributors
- **Tung Nguyen** - [GitHub](https://github.com/tungnd237)

