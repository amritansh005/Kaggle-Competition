import ollama
import json
from typing import Dict, List, Any

class MathQueryProcessor:
    def __init__(self):
        self.client = ollama.Client()
        self.model = "gemma3n:e4b-it-fp16"
        
    def parse_query(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        Parse this mathematical query and extract:
        1. Mathematical operations (plot, integrate, derive, solve, etc.)
        2. Functions/expressions
        3. Variables and their ranges
        4. Additional requirements (tangent lines, critical points, etc.)
        5. Whether this requires advanced features (complex analysis, DEs, etc.)
        
        Query: {query}
        
        Return as JSON with structure:
        {{
            "operations": [],
            "expressions": [],
            "variables": {{}},
            "special_features": [],
            "advanced_feature": null or string
        }}
        """
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            format="json"
        )
        
        return json.loads(response['response'])