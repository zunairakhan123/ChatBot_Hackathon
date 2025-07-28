# chat_manager.py - Chat Session Management System
import json
import os
import uuid
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

@dataclass
class ChatMessage:
    """Individual chat message structure"""
    message_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    rating: Optional[int] = None  # 1 for thumbs up, -1 for thumbs down, None for no rating
    is_bookmarked: bool = False
    source_documents: List[str] = None
    
    def __post_init__(self):
        if self.source_documents is None:
            self.source_documents = []

@dataclass
class ChatSession:
    """Chat session data structure"""
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = None
    is_archived: bool = False
    tags: List[str] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.tags is None:
            self.tags = []

class ChatManager:
    """Manages chat sessions and messages"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.sessions_file = self.data_dir / "sessions.json"
        self.ensure_sessions_file()
    
    def ensure_sessions_file(self):
        """Ensure sessions file exists"""
        if not self.sessions_file.exists():
            with open(self.sessions_file, 'w') as f:
                json.dump({}, f)
    
    def create_session(self, user_id: str, title: str = None) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            title=title,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        try:
            sessions = self.load_all_sessions()
            sessions[session_id] = asdict(session)
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            return session_id
            
        except Exception as e:
            raise Exception(f"Failed to create session: {str(e)}")
    
    def load_all_sessions(self) -> Dict[str, Dict]:
        """Load all sessions from storage"""
        try:
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID"""
        sessions = self.load_all_sessions()
        session_data = sessions.get(session_id)
        
        if session_data:
            # Convert message dictionaries back to ChatMessage objects
            messages = []
            for msg_data in session_data.get('messages', []):
                messages.append(ChatMessage(**msg_data))
            session_data['messages'] = messages
            return ChatSession(**session_data)
        return None
    
    def get_user_sessions(self, user_id: str, include_archived: bool = False) -> List[ChatSession]:
        """Get all sessions for a user"""
        sessions = self.load_all_sessions()
        user_sessions = []
        
        for session_data in sessions.values():
            if session_data.get('user_id') == user_id:
                if include_archived or not session_data.get('is_archived', False):
                    # Convert message dictionaries back to ChatMessage objects
                    messages = []
                    for msg_data in session_data.get('messages', []):
                        messages.append(ChatMessage(**msg_data))
                    session_data['messages'] = messages
                    user_sessions.append(ChatSession(**session_data))
        
        # Sort by updated_at descending
        user_sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return user_sessions
    
    def add_message(self, session_id: str, role: str, content: str, source_documents: List[str] = None) -> str:
        """Add a message to a chat session"""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        message = ChatMessage(
            message_id=message_id,
            role=role,
            content=content,
            timestamp=timestamp,
            source_documents=source_documents or []
        )
        
        try:
            sessions = self.load_all_sessions()
            
            if session_id not in sessions:
                raise ValueError(f"Session {session_id} not found")
            
            # Convert message to dict for storage
            message_dict = asdict(message)
            sessions[session_id]['messages'].append(message_dict)
            sessions[session_id]['updated_at'] = timestamp
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            return message_id
            
        except Exception as e:
            raise Exception(f"Failed to add message: {str(e)}")
    
    def rate_message(self, session_id: str, message_id: str, rating: int) -> bool:
        """Rate a message (1 for thumbs up, -1 for thumbs down)"""
        try:
            sessions = self.load_all_sessions()
            
            if session_id not in sessions:
                return False
            
            for message in sessions[session_id]['messages']:
                if message['message_id'] == message_id:
                    message['rating'] = rating
                    sessions[session_id]['updated_at'] = datetime.now().isoformat()
                    
                    with open(self.sessions_file, 'w') as f:
                        json.dump(sessions, f, indent=2)
                    
                    return True
            
            return False
            
        except Exception:
            return False
    
    def bookmark_message(self, session_id: str, message_id: str, is_bookmarked: bool = True) -> bool:
        """Bookmark or unbookmark a message"""
        try:
            sessions = self.load_all_sessions()
            
            if session_id not in sessions:
                return False
            
            for message in sessions[session_id]['messages']:
                if message['message_id'] == message_id:
                    message['is_bookmarked'] = is_bookmarked
                    sessions[session_id]['updated_at'] = datetime.now().isoformat()
                    
                    with open(self.sessions_file, 'w') as f:
                        json.dump(sessions, f, indent=2)
                    
                    return True
            
            return False
            
        except Exception:
            return False
    
    def get_bookmarked_messages(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all bookmarked messages for a user"""
        sessions = self.load_all_sessions()
        bookmarked = []
        
        for session_data in sessions.values():
            if session_data.get('user_id') == user_id:
                for message in session_data.get('messages', []):
                    if message.get('is_bookmarked', False):
                        bookmarked.append({
                            'session_id': session_data['session_id'],
                            'session_title': session_data['title'],
                            'message': message,
                            'timestamp': message['timestamp']
                        })
        
        # Sort by timestamp descending
        bookmarked.sort(key=lambda x: x['timestamp'], reverse=True)
        return bookmarked
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title"""
        try:
            sessions = self.load_all_sessions()
            
            if session_id not in sessions:
                return False
            
            sessions[session_id]['title'] = title
            sessions[session_id]['updated_at'] = datetime.now().isoformat()
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            return True
            
        except Exception:
            return False
    
    def archive_session(self, session_id: str, is_archived: bool = True) -> bool:
        """Archive or unarchive a session"""
        try:
            sessions = self.load_all_sessions()
            
            if session_id not in sessions:
                return False
            
            sessions[session_id]['is_archived'] = is_archived
            sessions[session_id]['updated_at'] = datetime.now().isoformat()
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            return True
            
        except Exception:
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        try:
            sessions = self.load_all_sessions()
            
            if session_id in sessions:
                del sessions[session_id]
                
                with open(self.sessions_file, 'w') as f:
                    json.dump(sessions, f, indent=2)
                
                return True
            return False
            
        except Exception:
            return False
    
    def export_chat_history(self, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """Export chat history for a user or specific session"""
        if session_id:
            session = self.get_session(session_id)
            if session and session.user_id == user_id:
                return {
                    'export_type': 'single_session',
                    'session': asdict(session),
                    'exported_at': datetime.now().isoformat()
                }
        else:
            sessions = self.get_user_sessions(user_id, include_archived=True)
            return {
                'export_type': 'all_sessions',
                'sessions': [asdict(session) for session in sessions],
                'exported_at': datetime.now().isoformat(),
                'total_sessions': len(sessions)
            }
        
        return {}
    
    def get_chat_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get chat statistics for a user"""
        sessions = self.get_user_sessions(user_id, include_archived=True)
        
        total_messages = 0
        total_user_messages = 0
        total_assistant_messages = 0
        bookmarked_count = 0
        rated_messages = {'positive': 0, 'negative': 0}
        
        for session in sessions:
            total_messages += len(session.messages)
            for message in session.messages:
                if message.role == 'user':
                    total_user_messages += 1
                else:
                    total_assistant_messages += 1
                
                if message.is_bookmarked:
                    bookmarked_count += 1
                
                if message.rating == 1:
                    rated_messages['positive'] += 1
                elif message.rating == -1:
                    rated_messages['negative'] += 1
        
        return {
            'total_sessions': len(sessions),
            'total_messages': total_messages,
            'user_messages': total_user_messages,
            'assistant_messages': total_assistant_messages,
            'bookmarked_messages': bookmarked_count,
            'message_ratings': rated_messages,
            'average_messages_per_session': total_messages / len(sessions) if sessions else 0
        }
