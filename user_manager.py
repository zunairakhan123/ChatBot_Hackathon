# user_manager.py - User Profile Management System
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: str
    username: str
    display_name: str
    expertise_level: str = "Beginner"
    preferred_languages: List[str] = None
    learning_goals: str = ""
    created_at: str = None
    last_active: str = None
    total_chats: int = 0
    favorite_responses: List[str] = None
    theme_preference: str = "light"
    response_format: str = "detailed"  # detailed, concise, bullet_points
    
    def __post_init__(self):
        if self.preferred_languages is None:
            self.preferred_languages = []
        if self.favorite_responses is None:
            self.favorite_responses = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_active is None:
            self.last_active = datetime.now().isoformat()

class UserManager:
    """Manages user profiles and preferences"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.users_file = self.data_dir / "users.json"
        self.ensure_users_file()
    
    def ensure_users_file(self):
        """Ensure users file exists"""
        if not self.users_file.exists():
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def create_user(self, profile: UserProfile) -> bool:
        """Create a new user profile"""
        try:
            users = self.load_all_users()
            
            # Check if username already exists
            for user_data in users.values():
                if user_data.get('username') == profile.username:
                    raise ValueError(f"Username '{profile.username}' already exists")
            
            # Save user profile
            users[profile.user_id] = asdict(profile)
            
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to create user: {str(e)}")
    
    def load_all_users(self) -> Dict[str, Dict]:
        """Load all users from storage"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        users = self.load_all_users()
        user_data = users.get(user_id)
        
        if user_data:
            return UserProfile(**user_data)
        return None
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get user profile by username"""
        users = self.load_all_users()
        
        for user_data in users.values():
            if user_data.get('username') == username:
                return UserProfile(**user_data)
        return None
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            users = self.load_all_users()
            
            if user_id not in users:
                return False
            
            # Update fields
            users[user_id].update(updates)
            users[user_id]['last_active'] = datetime.now().isoformat()
            
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return True
            
        except Exception:
            return False
    
    def increment_chat_count(self, user_id: str):
        """Increment user's total chat count"""
        self.update_user(user_id, {'total_chats': self.get_chat_count(user_id) + 1})
    
    def get_chat_count(self, user_id: str) -> int:
        """Get user's total chat count"""
        user = self.get_user(user_id)
        return user.total_chats if user else 0
    
    def add_favorite_response(self, user_id: str, response_id: str) -> bool:
        """Add response to user's favorites"""
        user = self.get_user(user_id)
        if user and response_id not in user.favorite_responses:
            user.favorite_responses.append(response_id)
            return self.update_user(user_id, {'favorite_responses': user.favorite_responses})
        return False
    
    def remove_favorite_response(self, user_id: str, response_id: str) -> bool:
        """Remove response from user's favorites"""
        user = self.get_user(user_id)
        if user and response_id in user.favorite_responses:
            user.favorite_responses.remove(response_id)
            return self.update_user(user_id, {'favorite_responses': user.favorite_responses})
        return False
    
    def is_favorite_response(self, user_id: str, response_id: str) -> bool:
        """Check if response is in user's favorites"""
        user = self.get_user(user_id)
        return user and response_id in user.favorite_responses
    
    def get_all_usernames(self) -> List[str]:
        """Get list of all usernames"""
        users = self.load_all_users()
        return [user_data.get('username', '') for user_data in users.values() if user_data.get('username')]
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user profile"""
        try:
            users = self.load_all_users()
            
            if user_id in users:
                del users[user_id]
                
                with open(self.users_file, 'w') as f:
                    json.dump(users, f, indent=2)
                
                return True
            return False
            
        except Exception:
            return False
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        user = self.get_user(user_id)
        if not user:
            return {}
        
        return {
            'total_chats': user.total_chats,
            'favorite_count': len(user.favorite_responses),
            'member_since': user.created_at,
            'last_active': user.last_active,
            'expertise_level': user.expertise_level,
            'preferred_languages': user.preferred_languages
        }
