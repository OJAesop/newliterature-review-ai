from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import datetime, timedelta
import os
from functools import wraps
import re

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///litreview.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
CORS(app, supports_credentials=True)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    organization = db.Column(db.String(100), nullable=True)
    research_areas = db.Column(db.Text, nullable=True)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    subscription_tier = db.Column(db.String(20), default='free')  # free, pro, enterprise
    
    # Relationships
    literature_reviews = db.relationship('LiteratureReview', backref='author', lazy=True)

class LiteratureReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    research_query = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='draft')  # draft, processing, completed, error
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Results storage (would typically be in separate tables)
    papers_found = db.Column(db.Integer, default=0)
    summary_generated = db.Column(db.Boolean, default=False)

# Utility functions
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    # At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

# Auth Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validation
        required_fields = ['email', 'password', 'full_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        email = data['email'].lower().strip()
        password = data['password']
        full_name = data['full_name'].strip()
        
        # Validate email
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password
        is_valid, message = validate_password(password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'User already exists with this email'}), 409
        
        # Create new user
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            organization=data.get('organization', ''),
            research_areas=data.get('research_areas', '')
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Create access token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'organization': user.organization,
                'subscription_tier': user.subscription_tier,
                'created_at': user.created_at.isoformat()
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.query.filter_by(email=email).first()
        
        if not user or not bcrypt.check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Create access token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'organization': user.organization,
                'subscription_tier': user.subscription_tier,
                'created_at': user.created_at.isoformat()
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': {
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'organization': user.organization,
                'research_areas': user.research_areas,
                'subscription_tier': user.subscription_tier,
                'created_at': user.created_at.isoformat()
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get user information'}), 500

# Dashboard Routes
@app.route('/api/dashboard/stats', methods=['GET'])
@jwt_required()
def get_dashboard_stats():
    try:
        user_id = get_jwt_identity()
        
        # Get user's literature reviews
        reviews = LiteratureReview.query.filter_by(user_id=user_id).all()
        
        stats = {
            'total_reviews': len(reviews),
            'completed_reviews': len([r for r in reviews if r.status == 'completed']),
            'papers_analyzed': sum(r.papers_found for r in reviews),
            'recent_reviews': [
                {
                    'id': r.id,
                    'title': r.title,
                    'status': r.status,
                    'created_at': r.created_at.isoformat(),
                    'papers_found': r.papers_found
                }
                for r in sorted(reviews, key=lambda x: x.updated_at, reverse=True)[:5]
            ]
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get dashboard stats'}), 500

@app.route('/api/reviews', methods=['GET', 'POST'])
@jwt_required()
def handle_reviews():
    user_id = get_jwt_identity()
    
    if request.method == 'GET':
        try:
            reviews = LiteratureReview.query.filter_by(user_id=user_id).all()
            return jsonify({
                'reviews': [
                    {
                        'id': r.id,
                        'title': r.title,
                        'description': r.description,
                        'status': r.status,
                        'created_at': r.created_at.isoformat(),
                        'updated_at': r.updated_at.isoformat(),
                        'papers_found': r.papers_found
                    }
                    for r in reviews
                ]
            }), 200
        except Exception as e:
            return jsonify({'error': 'Failed to get reviews'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if not data.get('title') or not data.get('research_query'):
                return jsonify({'error': 'Title and research query are required'}), 400
            
            review = LiteratureReview(
                title=data['title'],
                description=data.get('description', ''),
                research_query=data['research_query'],
                user_id=user_id
            )
            
            db.session.add(review)
            db.session.commit()
            
            return jsonify({
                'message': 'Literature review created successfully',
                'review_id': review.id
            }), 201
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': 'Failed to create review'}), 500

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, port=5000)