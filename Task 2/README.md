# Task 2 – Support Ticket Classification & Prioritization

## 📌 Project Overview
This project implements an NLP-based system to automatically classify customer support tickets into appropriate departments and assign priority levels.  
The goal is to reduce manual effort in ticket triaging and improve response time in real business support environments.

---

## 🎯 Problem Statement
In real organizations, thousands of support tickets are received daily.  
Manually reading and routing them causes:

- Delay in response  
- Human errors  
- Increased workload on support teams  

This system automates:

1. **Ticket Category Classification**  
2. **Priority Assignment**  
3. **End-to-End Inference through UI**

---

## 🛠 Techniques Implemented

### 1. Classical Machine Learning Models
- Text representation using **TF–IDF (unigram + bigram)**
- Models trained:
  - Logistic Regression  
  - Linear SVM

These models capture keyword-level patterns and provide strong baseline performance.

### 2. Transformer Based Model
- **DistilBERT Fine Tuning**
- Contextual understanding of ticket descriptions
- Able to capture semantic meaning instead of only keywords

### 3. Priority Assignment
- Rule-based urgency detection using domain keywords  
- Labels: **High / Medium / Low**

### 4. Frontend Interface
- Streamlit application for real-time prediction

---

## 📂 Files in Repository

- `task2.ipynb` – Complete training pipeline  
- `app.py` – Streamlit frontend  
- `model.pt` – Trained DistilBERT model  
- `README.md` – Project documentation  

---

## 📊 Results

DistilBERT performed best due to:
- Contextual embeddings  
- Understanding sentence semantics  
- Better generalization across categories

---

## 🚀 System Output Example

**Input Ticket:**  
“my laptop is not working and system crashed”

**Output:**  
- Category → Hardware  
- Priority → High

---

## 🧠 Workflow

1. User submits ticket text  
2. Text is cleaned & tokenized  
3. DistilBERT predicts category  
4. Priority inferred from urgency  
5. Result displayed via UI

---

## 💼 Business Benefits

- Automatic ticket routing  
- Reduced manual triage time  
- Faster SLA resolution  
- Consistent categorization  
- Improved customer satisfaction

---

## ▶ How to Run

```bash
streamlit run app.py
