import ollama
from typing import Dict, Any

class ExplanationGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def generate_explanation(self, operation: str, input_data: Dict[str, Any], 
                           result: Dict[str, Any], visualization_data: Dict[str, Any]) -> str:
        prompt = f"""
        Explain this mathematical operation in clear, educational terms:
        
        Operation: {operation}
        Input: {input_data}
        
        Focus on:
        1. What the operation does conceptually
        2. Why the result makes sense
        3. How to interpret the visualization
        4. Real-world applications
        
        Keep it concise but insightful.
        """
        
        try:
            response = self.llm.generate(
                model="gemma3n:e4b-it-fp16",
                prompt=prompt
            )
            return response['response']
        except:
            return f"This {operation} operation processes the given mathematical expression and generates a visualization to help understand the concept better."