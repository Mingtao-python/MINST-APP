# AI Learning Journey: Weeks 5-11

This repository contains a comprehensive AI learning project spanning weeks 5 through 11 of an intensive AI course. The project progresses from fundamental data analysis to building a complete machine learning system for handwritten digit recognition, incorporating security considerations and best practices.

---

## 🎯 Project Overview

This repository documents a 7-week journey in AI development, starting with data exploration and culminating in a production-ready digit recognition system. Each week builds upon the previous, introducing new concepts and best practices in machine learning, software engineering, and cybersecurity.

**Main Achievements:**
- ✅ 4 foundational AI projects (Weeks 5-8)
- ✅ Complete ML pipeline implementation (Weeks 9-11)
- ✅ Modular code architecture
- ✅ Model evaluation and optimization
- ✅ Security-aware system design
- ✅ Error analysis and improvement strategies

---

## 📅 Weekly Breakdown

### Week 5: Data Understanding & Statistics
**Focus:** Data exploration and visualization
- Dataset analysis (Digits dataset)
- Statistical distributions
- Data visualization techniques
- Confusion analysis between similar digits

**Key Files:** `第5周/第5周任务.py`
**Skills Learned:** pandas, matplotlib, data statistics

### Week 6: Machine Learning Basics
**Focus:** Model training and evaluation
- Train/test split concepts
- Logistic Regression implementation
- Data preprocessing (Standardization)
- Model accuracy and confusion matrices

**Key Files:** `第6周/第6周任务.py`
**Skills Learned:** scikit-learn, model evaluation

### Week 7: Model Comparison
**Focus:** Algorithm comparison and analysis
- Multiple ML algorithms (Logistic Regression, KNN, Random Forest)
- Performance benchmarking
- Training time analysis
- Confusion matrix interpretation

**Key Files:** `第7周/第7周任务.py`
**Skills Learned:** Model selection, performance metrics

### Week 8: Dimensionality Reduction
**Focus:** PCA and high-dimensional data
- Principal Component Analysis (PCA)
- 2D visualization of high-dimensional data
- Dimensionality reduction concepts
- Feature importance analysis

**Key Files:** `第8周/第8周任务.py`
**Skills Learned:** PCA, data visualization

### Week 9: Complete ML System (Modular)
**Focus:** System architecture and modular design
- Modular code structure (src/ organization)
- Complete ML pipeline
- Error analysis and visualization
- Security basics: Input validation

**Key Files:** `第9周/main.py`, `第9周/src/`
**Skills Learned:** Software architecture, error analysis

### Week 10: Model Evaluation & Analysis
**Focus:** Advanced evaluation metrics
- Precision, Recall, F1-Score
- Classification reports
- Error sample analysis
- Security: SQL injection simulation

**Key Files:** `第10周/main.py`, `第10周/src/`
**Skills Learned:** Advanced metrics, security awareness

### Week 11: Model Optimization & Multi-Model System
**Focus:** Hyperparameter tuning and system integration
- Parameter optimization for multiple models
- Automated model selection
- Performance comparison
- Security: Vulnerability fixes

**Key Files:** `第11周/main.py`, `第11周/src/`
**Skills Learned:** Hyperparameter tuning, system design

## 🏗️ Project Structure

```
├── 第5周/                          # Week 5: Data Analysis
├── 第6周/                          # Week 6: Basic ML
├── 第7周/                          # Week 7: Model Comparison
├── 第8周/                          # Week 8: PCA & Dimensionality Reduction
├── 第9周/                          # Week 9: Modular ML System
│   ├── main.py
│   └── src/
├── 第10周/                         # Week 10: Advanced Evaluation
│   ├── main.py
│   └── src/
├── 第11周/                         # Week 11: Multi-Model System
│   ├── main.py
│   └── src/
├── 安全与攻击/                     # Security Modules
│   ├── 第9周/
│   ├── 第10周/
│   └── 第11周/
├── 学习计划/                       # Course Plans & Objectives
├── data/                           # Datasets
├── results/                        # Output results
└── README.md
```

---

## 📋 Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn (optional, for advanced visualizations)

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-learning-journey
   ```

2. **Install dependencies:**
   ```bash
   pip install scikit-learn pandas numpy matplotlib
   ```

3. **Run individual weeks:**
   ```bash
   # Week 5: Data Analysis
   python 第5周/第5周任务.py

   # Week 9: Modular System
   python 第9周/main.py

   # Week 11: Multi-Model System
   python 第11周/main.py
   ```

---

## 🎓 Key Learnings

### Technical Skills
- **Data Science:** Data loading, preprocessing, visualization
- **Machine Learning:** Model training, evaluation, optimization
- **Software Engineering:** Modular code design, error handling
- **Mathematics:** Statistical analysis, dimensionality reduction

### Problem-Solving
- Algorithm selection based on data characteristics
- Performance optimization through parameter tuning
- Error analysis and model improvement strategies
- Security-conscious programming practices

### System Design
- Building complete ML pipelines
- Modular architecture for maintainability
- Automated model selection systems
- User input validation and security

---

## 🔒 Security Considerations

Weeks 9-11 incorporate security awareness training:

- **Input Validation:** Protecting against malformed inputs
- **SQL Injection Prevention:** Parameterized queries
- **Error Handling:** Graceful failure management
- **System Hardening:** Defense against common attacks

**Important:** All security exercises are conducted in controlled, local environments only. No external systems are targeted.

---

**Course Progression:** This work represents a foundational AI learning journey, preparing for more advanced projects in computer vision, natural language processing, and production AI systems.
