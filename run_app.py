#!/usr/bin/env python3
"""
Startup script that fixes database issues and runs the application
"""
import os
import sys
import subprocess

def run_database_fix():
    """Run the database fix script"""
    print("Running database fix...")
    try:
        if os.path.exists('fix_database.py'):
            exec(open('fix_database.py').read())
            print("✓ Database fix completed")
        else:
            print("⚠ Database fix script not found")
    except Exception as e:
        print(f"⚠ Database fix error: {e}")

def run_app():
    """Run the Flask application"""
    print("Starting Flask application...")
    try:
        # Set environment variables
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = 'true'
        
        # Import and run the app
        from app import app
        
        print("✓ Application starting on http://127.0.0.1:5000")
        print("✓ Press Ctrl+C to stop the application")
        
        app.run(debug=True, host='127.0.0.1', port=5000)
        
    except KeyboardInterrupt:
        print("\n✓ Application stopped by user")
    except Exception as e:
        print(f"✗ Application error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Smart Project Allocation System ===")
    print("Initializing...")
    
    # Fix database first
    run_database_fix()
    
    # Then run the app
    run_app()