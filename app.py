import os
import sys
import subprocess
from flask import Flask, render_template, request
from source.info import get_monster_info

# Base directory (one level up from this file)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Flask app setup
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'wyvernaire-secret'

# Uploads configuration
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Predictor script path
PREDICTOR_SCRIPT = os.path.join(BASE_DIR, 'source', 'predictor.py')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    flash_message = None
    flash_type = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename:
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            command = [
                sys.executable,
                PREDICTOR_SCRIPT,
                "--image", save_path
            ]

            try:
                completed_process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                output = completed_process.stdout

                # Parse top-1 result
                result = parse_prediction_output(output, filename)

                if result:
                    flash_message = "Prediction successful!"
                    flash_type = "success"
                else:
                    flash_message = "Prediction output could not be parsed."
                    flash_type = "danger"

            except subprocess.CalledProcessError:
                flash_message = "Prediction script failed."
                flash_type = "danger"
        else:
            flash_message = "No file selected."
            flash_type = "danger"

    return render_template(
        'index.html',
        result=result,
        flash_message=flash_message,
        flash_type=flash_type
    )

def parse_prediction_output(output, filename):
    """Parses the prediction output from predictor.py."""
    lines = output.splitlines()
    for line in lines:
        if line.strip().startswith("1. "):
            parts = line.strip().split(" - ")
            if len(parts) == 2:
                raw_monster_name = parts[0][3:].strip()
                confidence_str = parts[1].replace("confidence", "").replace("%", "").strip()

                try:
                    confidence = float(confidence_str) / 100
                except ValueError:
                    return None

                display_monster_name = raw_monster_name.replace('_', ' ')
                monster_info = get_monster_info(raw_monster_name)

                return {
                    "monster": display_monster_name,
                    "confidence": confidence,
                    "image_path": f"uploads/{filename}",
                    "info": monster_info
                }
    return None

if __name__ == '__main__':
    app.run(debug=True)
