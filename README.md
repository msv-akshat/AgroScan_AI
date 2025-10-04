# AgroScan AI

**AI-Powered Plant Disease Detection — Serverless, Cloud-Native, Production-Ready**

AgroScan AI is a robust web platform that leverages deep learning (CNN) and AWS serverless architecture for instant crop disease diagnosis. Farmers and researchers can get rapid, reliable results through an integrated React frontend and scalable FastAPI backend.

***

## Table of Contents

1. [Live Demo](#live-demo)
2. [Features](#features)
3. [Screenshots](#screenshots)
4. [Architecture](#architecture)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Setup \& Deployment](#setup--deployment)
8. [Model Training](#model-training)
9. [Roadmap](#roadmap)
10. [Author](#author)
11. [License](#license)

***

## Live Demo

- **[Try AgroScan AI](https://main.d3n8iyuxmo9pz7.amplifyapp.com/)**
- **[GitHub Repo](https://github.com/msv-akshat/AgroScan_AI)**

***

## Features

- Fast, accurate crop disease detection using deep learning
- Supports maize, tomato, apple, grape, and 30+ crop diseases
- Serverless backend (FastAPI, Lambda, Docker, ECR, API Gateway)
- Responsive React + Tailwind CSS frontend (AWS Amplify hosting)
- Easy model retraining, auto deployments via GitHub Actions
- Architecture designed for robust scaling and rapid inference

***

## Screenshots

**User Interface**

**Detection Results**

**Accuracy Curve**

***

## Architecture

*AWS Amplify for frontend, API Gateway for secure routing, Lambda (ECR/Docker) running FastAPI, and modular cloud-native flow.*

***

## Tech Stack

- **Frontend:** React.js, Tailwind CSS, AWS Amplify
- **Backend:** FastAPI (Python), Docker, AWS Lambda, AWS API Gateway, AWS ECR
- **ML Model:** TensorFlow, Keras, PyTorch (training notebook included), OpenCV
- **CI/CD \& Infra:** GitHub Actions, AWS CloudWatch

***

## Project Structure

```
AgroScan_AI/
├── frontend/        # React application
│   └── src/components/
│   └── README.md
├── ml_model/        # Backend (FastAPI, Lambda)
│   └── notebooks/   # Model training notebook
│   └── Dockerfile, app_lambda.py, etc.
└── README.md
```


***

## Setup \& Deployment

### 1. Clone

```bash
git clone https://github.com/msv-akshat/AgroScan_AI.git
cd AgroScan_AI
```


### 2. Frontend Setup

```bash
cd frontend
npm install
npm run build
# Deploy: connect to AWS Amplify, auto-deploy from GitHub main
```


### 3. Backend Setup

```bash
cd ml_model
pip install -r requirements.txt
docker build -t agroscan-ai .
# Push to AWS ECR:
# aws ecr get-login-password | docker login --username AWS --password-stdin
# docker push <your-ecr-url>/agroscan-ai:latest
# Deploy the container to AWS Lambda and connect via API Gateway
```


***

## Model Training

- Model code \& full training workflow in:

```
ml_model/notebooks/model_training_notebook.ipynb
```

- CNN (TensorFlow/Keras) achieves ~92% validation accuracy. Includes data augmentation, normalization, 38 crop/disease classes.

***

## Roadmap

- [ ] Visual crop/symptom localization
- [ ] Weather-aware crop health forecasts
- [ ] Multilingual and voice assistant interface
- [ ] Progressive Web App, mobile UX
- [ ] Analytical dashboard for farmers \& researchers

***

## Author

**Akshat Madamsetty**
AI, Cloud \& Full Stack Developer

- [LinkedIn](https://linkedin.com/in/sai-venkat-akshat-madamsetty-b3242328b/)
- [GitHub](https://github.com/msv-akshat)

***

## License

MIT License

***


