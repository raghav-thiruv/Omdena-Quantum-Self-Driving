from flask import Flask, render_template, request, jsonify, make_response, session, Blueprint
from flask_cors import CORS
from backend.views.auth import auth_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = "Z3_avb3hTsAoK8Uw"
app.register_blueprint(auth_bp, url_prefix='/home')
CORS(app)
    
@app.route('/unprotected')
def unprotected():
    return jsonify({'message': 'Anyone can view this'})

@app.route('protected')
def protected():
    return jsonify('message': 'Only those with a valid tokens can see.')

## create a home
@app.route("/")
def home():
    if not session.get("logged_in"):
        return render_template("login.html")
    else:
        return "Logged in currently"
    
if __name__ == "__main__":
    app.run(debug=True)