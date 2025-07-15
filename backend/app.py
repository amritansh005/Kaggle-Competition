#!/usr/bin/env python3.12
"""
Advanced Mathematical Visualization System
Main Application Entry Point
"""

import os
import json
import logging
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import subprocess
from flask_cors import CORS
from dotenv import load_dotenv
import ollama
import numpy as np
import sympy as sp
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.io as pio

# Import all our custom modules
from math_query_processor import MathQueryProcessor
from math_intent_parser import MathIntentParser
from computation_engine import ComputationEngine
from advanced_visualizer import AdvancedVisualizer
from explanation_generator import ExplanationGenerator
from advanced_features import MathVisualizationEngine

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['OUTPUT_FOLDER'] = os.getenv('OUTPUT_FOLDER', 'static/outputs')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

CORS(app)

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('cache', exist_ok=True)


class MathVisualizationSystem:
    """Main system controller"""
    
    def __init__(self):
        logger.info("Initializing Math Visualization System...")
        
        # Initialize components
        self.query_processor = MathQueryProcessor()
        self.intent_parser = MathIntentParser()
        self.computation_engine = ComputationEngine()
        self.visualizer = AdvancedVisualizer()
        self.advanced_engine = MathVisualizationEngine()
        self.explanation_generator = ExplanationGenerator(ollama.Client())
        
        logger.info("System initialized successfully")
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user's mathematical query"""
        try:
            logger.info(f"Processing query: {user_query}")
            
            # 1. Parse natural language query
            parsed_query = self.query_processor.parse_query(user_query)
            logger.debug(f"Parsed query: {parsed_query}")
            
            # 2. Extract mathematical intent (generalized)
            math_intent = self.intent_parser.process(parsed_query)
            logger.debug(f"Mathematical intent: {math_intent}")
            
            # 3. Route to appropriate processor
            if math_intent.get('advanced_feature'):
                # Use advanced features
                result = self.advanced_engine.process_advanced_query(
                    math_intent['advanced_feature'],
                    {
                        "operations": math_intent.get("operations", []),
                        "expressions": math_intent.get("expressions", []),
                        "variables": math_intent.get("variables", {}),
                        "features": math_intent.get("features", [])
                    }
                )
            else:
                # Standard processing (generalized)
                computation_results = self.computation_engine.compute(
                    math_intent.get("operations", []),
                    math_intent.get("expressions", []),
                    math_intent.get("variables", {}),
                    math_intent.get("features", [])
                )
                
                visualizations = self.visualizer.create_visualizations(
                    math_intent,
                    computation_results
                )
                
                result = {
                    'computation': computation_results,
                    'visualizations': visualizations
                }
            
            # 4. Generate explanation (generalized for all operations)
            explanation = self.explanation_generator.generate_explanation(
                ", ".join(math_intent.get("operations", [])),
                {
                    "expressions": [str(e) for e in math_intent.get("expressions", [])],
                    "variables": math_intent.get("variables", {}),
                    "features": math_intent.get("features", [])
                },
                result,
                {}
            )

            # Convert explanation markdown to HTML (backend-side)
            try:
                import markdown
                explanation_html = markdown.markdown(explanation, extensions=['extra', 'sane_lists'])
            except Exception:
                explanation_html = explanation  # fallback to plain text

            # 5. Prepare response
            response = {
                'success': True,
                'query': user_query,
                'intent': math_intent,
                'results': result,
                'explanation': explanation_html,
                'visualizations': self._prepare_visualizations_for_response(result)
            }

            # Recursively convert all SymPy objects to strings for JSON serialization
            def sympy_to_str(obj):
                import sympy
                import plotly.graph_objects as go
                if isinstance(obj, sympy.Basic):
                    return str(obj)
                elif isinstance(obj, go.Figure):
                    return obj.to_json()
                elif isinstance(obj, dict):
                    return {k: sympy_to_str(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sympy_to_str(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(sympy_to_str(v) for v in obj)
                else:
                    return obj

            # Recursively convert all SymPy objects to strings for JSON serialization
            response = sympy_to_str(response)

            # Ensure JSON serializability for all response fields
            import json
            response = json.loads(json.dumps(response))

            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'query': user_query
            }
    
    def _sanitize_plotly_dict(self, obj):
        """Recursively convert SymPy objects and numpy arrays in a dict to native Python types for JSON serialization."""
        import sympy
        import numpy as np
        if isinstance(obj, sympy.Basic):
            try:
                return float(obj)
            except Exception:
                return str(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize_plotly_dict(obj.tolist())
        elif isinstance(obj, dict):
            return {k: self._sanitize_plotly_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_plotly_dict(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_plotly_dict(v) for v in obj)
        else:
            return obj

    def _prepare_visualizations_for_response(self, result: Dict) -> List[Dict]:
        """Convert Plotly figures to JSON for response, sanitizing SymPy objects."""
        visualizations = []

        if 'visualizations' in result:
            for viz in result['visualizations']:
                if isinstance(viz, go.Figure):
                    fig_dict = viz.to_plotly_json()
                    sanitized_dict = self._sanitize_plotly_dict(fig_dict)
                    import json
                    visualizations.append({
                        'type': 'plotly',
                        'data': json.dumps(sanitized_dict)
                    })
                elif isinstance(viz, dict) and 'figure' in viz:
                    fig_dict = viz['figure'].to_plotly_json()
                    sanitized_dict = self._sanitize_plotly_dict(fig_dict)
                    import json
                    visualizations.append({
                        'type': 'plotly',
                        'data': json.dumps(sanitized_dict),
                        'metadata': {k: v for k, v in viz.items() if k != 'figure'}
                    })

        # Handle advanced feature results
        if 'result' in result and isinstance(result['result'], go.Figure):
            fig_dict = result['result'].to_plotly_json()
            sanitized_dict = self._sanitize_plotly_dict(fig_dict)
            import json
            visualizations.append({
                'type': 'plotly',
                'data': json.dumps(sanitized_dict)
            })
        elif 'result' in result and isinstance(result['result'], dict):
            if 'figure' in result['result']:
                fig_dict = result['result']['figure'].to_plotly_json()
                sanitized_dict = self._sanitize_plotly_dict(fig_dict)
                import json
                visualizations.append({
                    'type': 'plotly',
                    'data': json.dumps(sanitized_dict),
                    'metadata': {k: v for k, v in result['result'].items() if k != 'figure'}
                })

        return visualizations


# Initialize system
math_system = MathVisualizationSystem()


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_math_query():
    """API endpoint for processing mathematical queries"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'No query provided'
            }), 400
        
        # Process the query
        result = math_system.process_query(query)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example queries"""
    examples = [
        {
            'category': 'Calculus',
            'queries': [
                "Plot x³ - 3x² + 2 and show its critical points and inflection points",
                "Find and visualize the area between sin(x) and cos(x) from 0 to 2π",
                "Show the tangent line to e^x at x = 1",
                "Visualize the limit of sin(x)/x as x approaches 0"
            ]
        },
        {
            'category': '3D Visualization',
            'queries': [
                "Plot z = x² - y² and show its saddle point",
                "Visualize the surface x² + y² + z² = 1",
                "Show the intersection of z = x² + y² and z = 2 - x - y"
            ]
        },
        {
            'category': 'Differential Equations',
            'queries': [
                "Solve and plot y'' + 2y' + y = 0 with y(0)=1, y'(0)=0",
                "Show the phase portrait of the predator-prey equations",
                "Visualize the heat equation solution with initial temperature spike"
            ]
        },
        {
            'category': 'Complex Analysis',
            'queries': [
                "Visualize f(z) = z² + 1 in the complex plane",
                "Plot the Riemann surface of sqrt(z)",
                "Show the complex mapping w = e^z"
            ]
        },
        {
            'category': 'Linear Algebra',
            'queries': [
                "Show the transformation matrix [[2,1],[1,2]] acting on the unit circle",
                "Visualize eigenvalues and eigenvectors of [[3,1],[1,3]]",
                "Demonstrate rotation matrix for 45 degrees"
            ]
        },
        {
            'category': 'Statistics',
            'queries': [
                "Generate a normal distribution with μ=100, σ=15 and test for normality",
                "Perform linear regression on random data with visualization",
                "Show confidence intervals for sample mean"
            ]
        },
        {
            'category': 'Vector Fields',
            'queries': [
                "Plot the vector field F(x,y) = (-y, x) with streamlines",
                "Visualize the gradient field of f(x,y) = x² + y²",
                "Show the 3D vector field F = (y, -x, z)"
            ]
        },
        {
            'category': 'Parametric Equations',
            'queries': [
                "Plot the parametric curve x=cos(t), y=sin(2t) for t from 0 to 2π",
                "Visualize the 3D helix x=cos(t), y=sin(t), z=t",
                "Show velocity and acceleration for the cycloid"
            ]
        }
    ]
    
    return jsonify(examples)


@app.route('/api/handwriting', methods=['POST'])
def handwriting_to_text():
    """
    API endpoint for handwriting recognition from uploaded images.
    Uses LaTeX-OCR for math, falls back to EasyOCR for general text.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # Use Pix2tex CLI for math handwriting OCR
        try:
            result = subprocess.run(
                [os.path.join(app.root_path, "math_viz_env", "Scripts", "pix2tex.exe"), upload_path],
                capture_output=True, text=True, timeout=60
            )
            recognized_text = result.stdout.strip()
            if recognized_text:
                return jsonify({'success': True, 'text': recognized_text})
            else:
                return jsonify({'success': False, 'error': 'No text recognized by Pix2tex'}), 500
        except Exception as e:
            logger.error(f"Pix2tex OCR failed: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': 'Pix2tex OCR failed'}), 500

    except Exception as e:
        logger.error(f"Handwriting OCR error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'OCR failed'}), 500


@app.route('/api/export/<format>', methods=['POST'])
def export_visualization(format):
    """Export visualization in different formats"""
    try:
        data = request.get_json()
        figure_json = data.get('figure')
        
        if not figure_json:
            return jsonify({'error': 'No figure data provided'}), 400
        
        # Recreate figure from JSON
        fig = pio.from_json(figure_json)
        
        # Generate filename
        import uuid
        filename = f"math_viz_{uuid.uuid4().hex[:8]}"
        
        if format == 'png':
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}.png")
            fig.write_image(filepath, width=1200, height=800)
        elif format == 'svg':
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}.svg")
            fig.write_image(filepath)
        elif format == 'html':
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}.html")
            fig.write_html(filepath, include_plotlyjs='cdn')
        elif format == 'pdf':
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}.pdf")
            fig.write_image(filepath, format='pdf', width=1200, height=800)
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Export failed'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama connection
        client = ollama.Client()
        models = client.list()
        has_gemma = any('gemma3n' in model['name'] for model in models['models'])
        
        return jsonify({
            'status': 'healthy',
            'ollama': True,
            'gemma_model': has_gemma
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv('FLASK_ENV') == 'development'
    )
