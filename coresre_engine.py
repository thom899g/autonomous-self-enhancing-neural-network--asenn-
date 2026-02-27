"""
Self-Representation Engine (SRE) - Core component for maintaining causal graph
of network architecture, parameters, and performance history.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, validator

from config.firebase_config import firebase_manager

logger = logging.getLogger(__name__)

class NodeMetadata(BaseModel):
    """Metadata for individual neural network nodes"""
    node_id: str
    node_type: str = "layer"  # layer, activation, connection, etc.
    parameters: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    modified_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('node_type')
    def validate_node_type(cls, v):
        valid_types = {'layer', 'activation', 'connection', 'input', 'output', 'attention'}
        if v not in valid_types:
            raise ValueError(f"Node type must be one of {valid_types}")
        return v
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics and modification timestamp"""
        self.performance_metrics.update(metrics)
        self.modified_at = datetime.utcnow()

class EdgeMetadata(BaseModel):
    """Metadata for connections between nodes"""
    edge_id: str
    source_node: str
    target_node: str
    weight_matrix: Optional[np.ndarray] = None
    learning_rate: float = 0.001
    connection_strength: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_firestore_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-compatible dictionary"""
        data = self.dict(exclude={'weight_matrix'})
        if self.weight_matrix is not None:
            # Store as serialized bytes
            data['weight_matrix'] = self.weight_matrix.tobytes()
            data['weight_matrix_shape'] = self.weight_matrix.shape
            data['weight_matrix_dtype'] = str(self.weight_matrix.dtype)
        return data
    
    @classmethod
    def from_firestore_dict(cls,