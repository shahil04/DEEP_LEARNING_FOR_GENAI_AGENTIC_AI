Flask Face-Proctored MCQ Exam (Prototype)
========================================

Contents:
- app.py           : Flask backend (serves pages, receives frames, analyzes face/head-pose)
- templates/       : HTML templates (exam.html, report.html)
- static/js/camera.js : Frontend webcam capture & monitoring JS
- static/evidence/ : Folder where captured violation images are stored (created at runtime)
- requirements.txt : Python dependencies

How to run (local):
1. Create a virtual environment and activate it.
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows

2. Install requirements:
   pip install -r requirements.txt

3. Run the Flask app:
   python app.py

4. Open http://localhost:5000 in your browser. Allow webcam access when prompted.

Notes:
- This is a prototype/demo. For production use:
  * Use per-student sessions and persistent DB storage for violations.
  * Secure the app (HTTPS), authentication, CSRF protection.
  * Tune head-pose thresholds and model parameters for your environment.
  * Consider server-side performance: MediaPipe/OpenCV can be CPU intensive.
