# 1. Start with a base image
FROM python:3.9

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . .

# 5. Expose the port your Flask app will run on
EXPOSE 8080

# 6. Command to run your Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:application"]
