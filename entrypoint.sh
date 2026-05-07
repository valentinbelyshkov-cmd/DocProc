#!/bin/bash

# Create necessary directories
mkdir -p /app/uploads
chmod 777 /app/uploads
mkdir -p /app/debug_images
chmod 777 /app/debug_images

# Start the Flask application
exec python /app/app.py
