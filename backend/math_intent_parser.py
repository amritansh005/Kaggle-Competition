import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import re

class MathIntentParser:
    def __init__(self):
        pass
        
    def parse_expression(self, expr_str: str) -> sp.Expr:
        # Enhanced expression parsing with common notation support
        expr_str = self.preprocess_expression(expr_str)
        return parse_expr(expr_str, transformations='all')
    
    def preprocess_expression(self, expr: str) -> str:
        # Handle common mathematical notations
        replacements = {
            'sin²': 'sin**2',
            'cos²': 'cos**2',
            'e^': 'exp',
            '√': 'sqrt',
            '∫': 'integrate',
            '∂': 'diff',
            'π': 'pi',
            '∞': 'oo'
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        return expr

    def process(self, parsed_query: dict) -> dict:
        # Generalized: pass through all operations and expressions
        intent = {
            'operations': parsed_query.get('operations', []),
            'expressions': [self.parse_expression(expr) for expr in parsed_query.get('expressions', [])],
            'variables': parsed_query.get('variables', {}),
            'features': parsed_query.get('special_features', []),
            'advanced_feature': parsed_query.get('advanced_feature')
        }
        return intent