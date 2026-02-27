"""
Firebase configuration and initialization module.
Handles secure connection to Firestore for persistent storage of architectural metadata.
"""
import os
import logging
from typing import Optional
from dataclasses import dataclass

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Client as FirestoreClient

logger = logging.getLogger(__name__)

@dataclass
class FirebaseConfig:
    """Configuration for Firebase connection"""
    service_account_path: str = "config/firebase-service-account.json"
    project_id: str = "asenn-system-v1"
    collection_name: str = "architecture_versions"
    
    def validate_path(self) -> bool:
        """Validate that the service account file exists"""
        if not os.path.exists(self.service_account_path):
            logger.error(f"Firebase service account not found at: {self.service_account_path}")
            return False
        return True

class FirebaseManager:
    """Manages Firebase Firestore connection with error handling and reconnection logic"""
    
    _instance: Optional['FirebaseManager'] = None
    _db: Optional[FirestoreClient] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[FirebaseConfig] = None):
        if not self._initialized:
            self.config = config or FirebaseConfig()
            self._initialized = True
            logger.info("FirebaseManager initialized")
    
    def initialize(self) -> bool:
        """
        Initialize Firebase connection with comprehensive error handling
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Check if already initialized
            if firebase_admin._apps:
                logger.info("Firebase already initialized, reusing existing connection")
                self._db = firestore.client()
                return True
            
            # Validate service account path
            if not self.config.validate_path():
                logger.error("Cannot initialize Firebase without valid service account")
                return False
            
            # Initialize Firebase
            cred = credentials.Certificate(self.config.service_account_path)
            firebase_admin.initialize_app(cred, {
                'projectId': self.config.project_id
            })
            
            self._db = firestore.client()
            logger.info(f"Firebase Firestore initialized for project: {self.config.project_id}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Service account file not found: {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid Firebase configuration: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Firebase initialization: {e}")
            return False
    
    @property
    def db(self) -> FirestoreClient:
        """Get Firestore client with lazy initialization"""
        if self._db is None:
            if not self.initialize():
                raise ConnectionError("Failed to initialize Firebase Firestore")
        return self._db
    
    def get_collection(self) -> firestore.CollectionReference:
        """Get the main collection reference for architecture storage"""
        return self.db.collection(self.config.collection_name)
    
    def close(self):
        """Clean up Firebase connection"""
        if self._db:
            self._db = None
        logger.info("Firebase connection closed")

# Global instance for easy access
firebase_manager = FirebaseManager()