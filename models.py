# ============================================================
# MODELS — SQLite Database Models via SQLAlchemy
# ============================================================

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """Attorney/user account."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    chats = db.relationship("ChatSession", backref="user", lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"


class ChatSession(db.Model):
    """A single analysis session (one PDF = one session)."""
    __tablename__ = "chat_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    session_name = db.Column(db.String(200), default="New Session")
    document_filename = db.Column(db.String(200), nullable=True)
    document_path = db.Column(db.String(500), nullable=True)
    vector_store_path = db.Column(db.String(500), nullable=True)
    text_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = db.relationship("ChatMessage", backref="session", lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession {self.session_name}>"


class ChatMessage(db.Model):
    """Individual messages within a session."""
    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "user" or "assistant"
    content = db.Column(db.Text, nullable=False)
    feature_type = db.Column(db.String(50), nullable=True)  # qa / risk / summary / missing
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChatMessage {self.role} in session {self.session_id}>"
    
    
class ContractComparison(db.Model):
    """Stores contract comparison sessions."""
    __tablename__ = "contract_comparisons"

    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    session_name        = db.Column(db.String(200), default="Comparison")
    
    # Document 1
    doc1_filename       = db.Column(db.String(200), nullable=True)
    doc1_path           = db.Column(db.String(500), nullable=True)
    doc1_text_path      = db.Column(db.String(500), nullable=True)
    
    # Document 2
    doc2_filename       = db.Column(db.String(200), nullable=True)
    doc2_path           = db.Column(db.String(500), nullable=True)
    doc2_text_path      = db.Column(db.String(500), nullable=True)
    
    # Results
    comparison_result   = db.Column(db.Text, nullable=True)
    
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ContractComparison {self.session_name}>"