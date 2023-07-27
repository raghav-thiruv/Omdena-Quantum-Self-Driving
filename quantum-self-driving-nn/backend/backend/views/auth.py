from flask import request, jsonify, make_response, Blueprint, current_app
from functools import wraps
from datetime import datetime, timedelta
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)
secret_key = current_app.config['SECRET_KEY']

def token_required(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        auth_header = request.header.get('Authorization')
        if auth_header:
            jwt = auth_header.split()[1]
            try:
                # payload is user information 
                payload = jwt.verify(jwt, secret_key)
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Unauthorised!'})
        return func(*args, **kwargs)
    return decorated

@auth_bp.route("/login", methods=["POST", "GET"])
def login():
    #  Get the user's credentials from the request
    username = request.form['username']
    password = request.form['password']

    # Look up the user in the database
    user = User.query.filter_by(username=username).first()

    # Check if the user exists and the password is correct
    if user and check_password_hash(user.password_hash, password):
        token = jwt.encode({
            'user': request.form['username'],
            'expiration': str(datetime.utcnow() + timedelta(seconds=120))    
        }),
        auth_bp.config['SECRET_KEY']
        return jsonify({'token': token.decode('utf-8')})
    else:
        return make_response('Unable to verify', 403, {'WWW-Authenticate': 'Basic-realm: Authentication Failed!'})
    
@auth_bp.route('/register', methods=['POST'])
def register():
    # Get the user's information from the request
    username = request.json.get('username')
    password = request.json.get('password')

    # Hash the user's password
    password_hash = generate_password_hash(password)

    # Create the user in the database
    # store the hashed password
    
    # Generate a JWT for the user
    token = jwt.encode({'username': username}, auth_bp.config['SECRET_KEY'], algorithm='HS256')

    # Return the JWT to the client
    return jsonify({'token': token.decode('utf-8')})