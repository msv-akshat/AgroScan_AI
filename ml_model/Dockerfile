# ----------------------
# Build Stage
# ----------------------
FROM public.ecr.aws/lambda/python:3.11 AS build

# Install Python dependencies into /opt/python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target /opt/python

# ----------------------
# Final Runtime Stage
# ----------------------
FROM public.ecr.aws/lambda/python:3.11

# Copy only the built dependencies
COPY --from=build /opt/python ${LAMBDA_TASK_ROOT}

# Copy application code
COPY app_lambda.py ${LAMBDA_TASK_ROOT}

# Copy model into /opt (read-only storage)
COPY models/pretrained_model.onnx /opt/models/pretrained_model.onnx

# Environment variable for model path
ENV ONNX_MODEL_PATH=/opt/models/pretrained_model.onnx

# Lambda entrypoint
CMD ["app_lambda.lambda_handler"]
