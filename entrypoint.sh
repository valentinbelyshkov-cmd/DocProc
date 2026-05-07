#!/bin/sh

# Create necessary directories
mkdir -p /app/uploads
chmod 777 /app/uploads
mkdir -p /app/debug_images
chmod 777 /app/debug_images

# Start the Flask application
# Use python3 to be more explicit for Debian-based images
exec python3 /app/app.py
