# Deployment Guide

## Local Development Setup

### 1. Prerequisites
```bash
# Python 3.8+ required
python --version

# Git (for cloning)
git --version
```

### 2. Project Setup
```bash
# Clone repository
git clone <repository-url>
cd egyptian-landmarks

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

Required environment variables:
- `GEMINI_API_KEY`: Get from Google AI Studio
- `ROBOFLOW_API_KEY`: Get from Roboflow (optional, only needed for data download)

### 4. Model Files
Create a `models/` directory and place your trained models:
```
models/
├── best.pt                                    # YOLO model
└── egyptian_landmarks_classification_model.pth  # Classification model
```

### 5. Run Streamlit App
```bash
streamlit run app.py
```

## Training Pipeline

### Full Pipeline
```bash
python main.py --mode full
```

### Individual Steps
```bash
# Download dataset
python main.py --mode download

# Process dataset
python main.py --mode process

# Train YOLO
python main.py --mode train_yolo

# Train classifier
python main.py --mode train_classifier
```

## Google Colab Deployment

### 1. Upload Files
Upload all `.py` files to Colab:
- `config.py`
- `data_downloader.py`
- `data_processor.py`
- `yolo_trainer.py`
- `classification_trainer.py`
- `utils.py`
- `main.py`
- `app.py`

### 2. Install Dependencies
```python
!pip install -r requirements.txt
```

### 3. Set Environment Variables
```python
import os
os.environ['GEMINI_API_KEY'] = 'your_api_key_here'
```

### 4. Run Training
```python
!python main.py --mode full
```

### 5. Run Streamlit App
```python
# Install tunnel for public access
!pip install pyngrok

# Run with tunnel
!streamlit run app.py & npx localtunnel --port 8501
```

## Docker Deployment

### 1. Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run
```bash
# Build image
docker build -t egyptian-landmarks .

# Run container
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key_here \
  -v $(pwd)/models:/app/models \
  egyptian-landmarks
```

## Cloud Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard:
   - `GEMINI_API_KEY`
4. Deploy automatically

### Heroku
1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create egyptian-landmarks-app
heroku config:set GEMINI_API_KEY=your_key_here
git push heroku main
```

### AWS/GCP/Azure
- Use container services (ECS, Cloud Run, Container Instances)
- Deploy Docker image
- Configure environment variables
- Set up load balancing and scaling

## Production Considerations

### 1. Security
- Use secrets management for API keys
- Enable HTTPS
- Implement rate limiting
- Add authentication if needed

### 2. Performance
- Optimize model loading (cache models)
- Use GPU instances for inference
- Implement image resizing limits
- Add caching for Gemini responses

### 3. Monitoring
- Add logging and metrics
- Monitor model performance
- Set up health checks
- Track usage analytics

### 4. Scaling
- Use multiple replicas
- Implement load balancing
- Consider async processing for heavy tasks
- Use CDN for static assets

## Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure models are in `models/` directory
   - Check file paths in config

2. **Gemini API errors**
   - Verify API key is correct
   - Check API quotas and limits
   - Handle network timeouts

3. **Memory issues**
   - Reduce batch sizes
   - Use CPU instead of GPU if needed
   - Optimize image preprocessing

4. **Streamlit deployment issues**
   - Check port configuration
   - Verify all dependencies are installed
   - Check logs for specific errors

### Performance Optimization
- Use model quantization
- Implement model caching
- Optimize image preprocessing pipeline
- Use async processing where possible