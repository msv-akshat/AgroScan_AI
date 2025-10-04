
# AgroScan AI

**AI-Powered Plant Disease Detection — Serverless, Cloud-Native, Production-Ready**

AgroScan AI leverages deep learning and AWS serverless architecture for instant crop disease diagnosis. Farmers and researchers get fast, reliable results through a React frontend, FastAPI backend, and automated cloud deployment.

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
9. [Planned Improvements](#planned-improvements)
10. [Author](#author)
11. [License](#license)

***

## Live Demo

- **[Try AgroScan AI](https://main.d3n8iyuxmo9pz7.amplifyapp.com/)**
- **[GitHub Repo](https://github.com/msv-akshat/AgroScan_AI)**

***

## Features

- Instant crop disease diagnosis from images
- 38+ supported crop-disease classes (maize, tomato, apple, grape, etc.)
- Serverless backend (FastAPI on AWS Lambda using Docker/ECR, API Gateway)
- Modern, responsive React + Tailwind CSS frontend (AWS Amplify)
- Easy retraining, modular infrastructure, seamless CI/CD via GitHub Actions
- Optimized for scalability and reliability

***

## Screenshots

**User Interface**
<img width="1918" height="1012" alt="Screenshot 2025-10-01 232836" src="https://github.com/user-attachments/assets/aaa36c1f-b36c-463e-850b-3161cd32ccc0" />

**Detection Results**
<img width="1913" height="1008" alt="Screenshot 2025-10-01 233020" src="https://github.com/user-attachments/assets/857c7f4b-b56c-4313-9f85-3d4339767b57" />

**Accuracy Curve**
<img width="1263" height="682" alt="Screenshot 2025-10-01 234333" src="https://github.com/user-attachments/assets/23a448fb-d7ab-4a5c-89e0-02dde545a1e6" />

***

## Architecture

<img width="1024" height="1024" alt="Gemini_Generated_Image_7q1le07q1le07q1l" src="https://github.com/user-attachments/assets/216825b3-38f8-45cf-b48f-3c483e6b4c6b" />
*AWS Amplify for frontend, API Gateway for secure routing, Lambda (ECR/Docker) running FastAPI, and modular, cloud-native design.*

***

## Tech Stack

| Layer | Technologies \& Tools |
| :-- | :-- |
| **Frontend** | React.js, Tailwind CSS, Vite, JavaScript, AWS Amplify |
| **Frontend Build** | Vite, npm, ESLint |
| **Backend** | FastAPI (Python), Docker, AWS Lambda, AWS API Gateway, AWS ECR |
| **ML Model** | TensorFlow, Keras, PyTorch, ONNX, OpenCV |
| **Infra / CI/CD** | AWS Amplify, Docker, GitHub Actions, Vercel (testing), AWS CloudWatch |


***

## Project Structure

```
AgroScan_AI/
├── .gitignore
├── README.md
├── package-lock.json
├── package.json
├── frontend/
│   ├── .gitignore
│   ├── README.md
│   ├── eslint.config.js
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── vercel.json
│   ├── vite.config.js
│   ├── public/
│   │   └── vite.svg
│   └── src/
│       ├── App.css
│       ├── App.jsx
│       ├── index.css
│       ├── main.jsx
│       ├── assets/
│       │   └── react.svg
│       ├── components/
│       │   ├── ImageUploader.jsx
│       │   ├── Loader.jsx
│       │   ├── PlantDiseasePredictor.jsx
│       │   ├── PlantSelector.jsx
│       │   ├── PredictionResult.jsx
│       │   └── TopKModal.jsx
│       └── config/
│           └── plants.js
├── ml_model/
│   ├── Dockerfile
│   ├── app_lambda.py
│   ├── check_model.py
│   ├── check_onnx.py
│   ├── require
│   ├── requirements-min.txt
│   ├── requirements.txt
│   ├── save_as_onnx.py
│   ├── temp.jpg
│   └── notebooks/
│       └── convert_to_tfjs.py
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
# Deploy to AWS Amplify for CI/CD (auto-deploy from GitHub main branch)
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

- Model training/execution workflow and code:

```
ml_model/notebooks/model_training_notebook.ipynb
```

- CNN (TensorFlow/Keras) achieves ~98% validation accuracy on 38+ classes with augmentation \& normalization.

***

## Planned Improvements

- Error handling for non-image/invalid file uploads
- CLI for simple model retraining and deployment
- Improved UI loading indicators and confidence explanations
- Expanded troubleshooting and FAQ in docs

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



