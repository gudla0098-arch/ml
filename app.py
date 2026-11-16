import os, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from model import predict_image, MODEL_PATH
BASE_DIR=Path(__file__).resolve().parent
UPLOAD_DIR=BASE_DIR/"uploads"; DB_PATH=BASE_DIR/"mediscan.db"
ALLOWED={"png","jpg","jpeg"}
app=Flask(__name__); app.config["MAX_CONTENT_LENGTH"]=10*1024*1024
UPLOAD_DIR.mkdir(exist_ok=True)
def db():
    conn=sqlite3.connect(DB_PATH); conn.row_factory=sqlite3.Row; return conn
def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, filename TEXT NOT NULL, prediction TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL, model_version TEXT)""")
def allowed_file(name): return "." in name and name.rsplit(".",1)[1].lower() in ALLOWED
@app.get("/")
def index(): return render_template("index.html")
@app.get("/api/health")
def health(): return jsonify({"status":"ok","model_ready":MODEL_PATH.exists(),"service":"MediScan AI"})
@app.get("/api/stats")
def stats():
    with db() as conn:
        total=conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        recent=conn.execute("SELECT id,filename,prediction,confidence,created_at FROM analyses ORDER BY created_at DESC LIMIT 6").fetchall()
        normal=conn.execute("SELECT COUNT(*) FROM analyses WHERE prediction='NORMAL'").fetchone()[0]
        pneumonia=conn.execute("SELECT COUNT(*) FROM analyses WHERE prediction='PNEUMONIA'").fetchone()[0]
    return jsonify({"total":total,"normal":normal,"pneumonia":pneumonia,"recent":[dict(x) for x in recent]})
@app.get("/api/history")
def history():
    with db() as conn: rows=conn.execute("SELECT id,filename,prediction,confidence,created_at FROM analyses ORDER BY created_at DESC LIMIT 100").fetchall()
    return jsonify([dict(x) for x in rows])
@app.post("/api/analyze")
def analyze():
    if "image" not in request.files: return jsonify({"error":"Choose a chest X-ray image first."}),400
    file=request.files["image"]
    if not file.filename: return jsonify({"error":"No file selected."}),400
    if not allowed_file(file.filename): return jsonify({"error":"Please upload a PNG or JPG image."}),415
    analysis_id=str(uuid.uuid4()); extension=secure_filename(file.filename).rsplit(".",1)[1].lower()
    path=UPLOAD_DIR/(analysis_id+"."+extension); file.save(path)
    try: result=predict_image(path)
    except RuntimeError as exc:
        path.unlink(missing_ok=True); return jsonify({"error":str(exc)}),503
    except Exception:
        path.unlink(missing_ok=True); app.logger.exception("Image analysis failed")
        return jsonify({"error":"The image could not be analyzed. Check that it is a valid chest X-ray."}),422
    created=datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute("INSERT INTO analyses VALUES (?,?,?,?,?,?)",(analysis_id,secure_filename(file.filename),result["prediction"],result["confidence"],created,result.get("model_version","1.0")))
    return jsonify({"id":analysis_id,"filename":secure_filename(file.filename),"prediction":result["prediction"],"confidence":result["confidence"],"created_at":created,"notice":"Educational AI output only. Not a medical diagnosis."})
@app.errorhandler(413)
def too_large(_): return jsonify({"error":"File is too large. Maximum upload size is 10 MB."}),413
@app.get("/uploads/<path:name>")
def uploaded(name): return send_from_directory(UPLOAD_DIR,name)
init_db()
if __name__=="__main__": app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=os.environ.get("FLASK_DEBUG")=="1")
