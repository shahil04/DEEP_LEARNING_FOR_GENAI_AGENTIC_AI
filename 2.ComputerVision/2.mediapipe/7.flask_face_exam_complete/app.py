from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import base64, os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "exam_secret_key_123"   # change in production
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "static", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                 max_num_faces=2,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

# Globals (prototype)
fault_count = 0
MAX_FAULTS = 10
# store violations as a list of dicts: {"file": relative_path, "reason": str}
violations = []

def save_violation(frame, reason):
    """Saves a violation image to static/evidence and records reason."""
    global violations
    filename = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)
    try:
        cv2.imwrite(filepath, frame)
        # Store path relative to project so templates can access via /static/...
        rel_path = os.path.join('static', 'evidence', filename)
        violations.append({"file": rel_path, "reason": reason})
    except Exception as e:
        print("Failed to save violation image:", e)

def get_head_pose(landmarks, frame_w, frame_h):
    """Estimate head pose (pitch, yaw, roll) using selected landmarks."""
    face_2d, face_3d = [], []
    # choose landmarks indices from MediaPipe face mesh
    # (These indices correspond to nose tip, eyes, mouth corners, chin-ish points)
    landmark_ids = [33, 263, 1, 61, 291, 199]
    for idx, lm in enumerate(landmarks.landmark):
        if idx in landmark_ids:
            x, y = int(lm.x * frame_w), int(lm.y * frame_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z * 3000])  # scale z to make solvePnP stable

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera internals
    focal_length = frame_w
    cam_matrix = np.array([[focal_length, 0, frame_w / 2],
                           [0, focal_length, frame_h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # SolvePnP
    try:
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        return pitch, yaw, roll
    except Exception as e:
        # If solvePnP fails, return zeros
        return 0.0, 0.0, 0.0

def analyze_frame(image_bgr):
    """Analyze a single BGR frame. Returns dict with status and faults."""
    global fault_count, violations
    frame_h, frame_w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "Face OK"

    if results.multi_face_landmarks:
        if len(results.multi_face_landmarks) > 1:
            status = "Multiple faces detected"
            fault_count += 1
            save_violation(image_bgr, status)
        else:
            # single face: check head pose
            for face_landmarks in results.multi_face_landmarks:
                pitch, yaw, roll = get_head_pose(face_landmarks, frame_w, frame_h)
                # thresholds (tweakable)
                if yaw > 20:
                    status = "Look into camera (Right)"
                    fault_count += 1
                    save_violation(image_bgr, status)
                elif yaw < -20:
                    status = "Look into camera (Left)"
                    fault_count += 1
                    save_violation(image_bgr, status)
                elif pitch > 15:
                    status = "Look into camera (Down)"
                    fault_count += 1
                    save_violation(image_bgr, status)
                elif pitch < -15:
                    status = "Look into camera (Up)"
                    fault_count += 1
                    save_violation(image_bgr, status)
                else:
                    status = "Face OK"
    else:
        status = "Face not detected"
        fault_count += 1
        save_violation(image_bgr, status)

    return {"status": status, "faults": fault_count, "max_faults": MAX_FAULTS}

@app.route('/')
def index():
    return render_template('exam.html')

@app.route('/monitor', methods=['POST'])
def monitor():
    """Receives base64 frame, analyzes it and returns JSON result."""
    try:
        data = request.json.get('frame', None)
        if not data:
            return jsonify({"error": "no frame data"}), 400

        # Decode base64 image
        header, encoded = data.split(',', 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = analyze_frame(frame)
        result['submit'] = True if result['faults'] >= MAX_FAULTS else False
        return jsonify(result)
    except Exception as e:
        print('Error in /monitor:', e)
        return jsonify({"error": str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit():
    """Handles exam submission form, computes score (example) and shows report."""
    # Simple scoring example (update as needed)
    score = 0
    if request.form.get('q1') == '4':
        score += 1
    if request.form.get('q2') == 'Delhi':
        score += 1

    # save to session for report rendering
    session['score'] = score
    session['violations'] = violations  # in production tie violations to user and persist in DB

    return redirect(url_for('report'))

@app.route('/report')
def report():
    score = session.get('score', 0)
    violations_session = session.get('violations', [])
    return render_template('report.html', score=score, violations=violations_session)

if __name__ == '__main__':
    # for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
