import sympy as sp
import numpy as np
from scipy import integrate, optimize
from typing import Dict, Any, List, Union

import base64
import io

# --- PDF Extraction Utility ---
def extract_text_and_images_from_pdf(pdf_base64: str):
    """
    Extracts text from a base64-encoded PDF.
    Returns: (text, page_offsets) where:
      - text: full extracted text
      - page_offsets: list of (start_offset, end_offset, page_num)
    """
    import fitz  # PyMuPDF
    pdf_bytes = base64.b64decode(pdf_base64)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    page_offsets = []
    offset = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        start_offset = offset
        full_text += page_text
        offset += len(page_text)
        end_offset = offset
        page_offsets.append((start_offset, end_offset, page_num + 1))
    return full_text, page_offsets

# --- Chapter-to-Micro-Lesson Compiler ---
def compile_micro_lessons(chapter_text: str, page_offsets: list = None) -> dict:
    """
    Segments, summarizes, and rewrites a chapter into micro-lessons using LLM.
    Handles large PDFs by chunking the text to fit the LLM's context window.
    Returns a dict: {micro_lessons: [...]}
    Each micro-lesson will include a "pages" field if page_offsets is provided.
    """
    import ollama
    import json

    CHUNK_SIZE = 10000
    OVERLAP = 2000  # 20% overlap for context preservation
    text_chunks = []
    chunk_ranges = []
    i = 0
    while i < len(chapter_text):
        text_chunks.append(chapter_text[i:i+CHUNK_SIZE])
        chunk_ranges.append((i, i+CHUNK_SIZE))
        i += CHUNK_SIZE - OVERLAP

    # Map each chunk to the PDF pages it covers
    chunk_pages = []
    if page_offsets:
        for start, end in chunk_ranges:
            pages = set()
            for p_start, p_end, page_num in page_offsets:
                # If the chunk overlaps with this page's text range
                if not (end <= p_start or start >= p_end):
                    pages.add(page_num)
            chunk_pages.append(sorted(pages))
    else:
        chunk_pages = [None] * len(text_chunks)

    all_micro_lessons = []
    client = ollama.Client()
    for idx, chunk in enumerate(text_chunks):
        prompt = f"""
You are an expert educational content compiler. Given the following textbook chapter segment, segment it into natural subtopics (using your own attention and topic boundaries), and for each subtopic, generate a micro-lesson in the following format:

- heading: (title of the subtopic)
- explanation: (clear, concise summary)
- analogy: (relatable analogy)
- example: (worked example)
- quizlet: (2-3 quick quiz questions)
- glossary: (important terms and definitions as a dictionary)
- emoji: (one or two relevant emoji for younger learners)

All output must be in valid JSON as a list of objects, one per micro-lesson. Do not include any extra text or commentary.

Chapter segment {idx+1}:
\"\"\"
{chunk}
\"\"\"
"""
        try:
            response = client.generate(
                model="gemma3n:e4b-it-fp16",
                prompt=prompt,
                stream=False,
                options={"temperature": 0.2, "max_tokens": 4096}
            )
            text = response["response"]
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                micro_lessons = json.loads(json_str)
            else:
                micro_lessons = json.loads(text)
            # Add page info to each micro-lesson in this chunk
            for ml in micro_lessons:
                if chunk_pages[idx]:
                    ml["pages"] = chunk_pages[idx]
            all_micro_lessons.extend(micro_lessons)
        except Exception as e:
            all_micro_lessons.append({
                "heading": f"Error in chunk {idx+1}",
                "explanation": f"LLM error: {str(e)}",
                "analogy": "",
                "example": "",
                "quizlet": [],
                "glossary": {},
                "emoji": "❌",
                "pages": chunk_pages[idx] if chunk_pages[idx] else []
            })
    return {
        "micro_lessons": all_micro_lessons
    }

class ComputationEngine:
    def __init__(self):
        self.symbolic_engine = SymbolicEngine()
        self.numerical_engine = NumericalEngine()

    def compute(self, operations: List[str], expressions: List[sp.Expr], variables: Dict[str, Any], features: List[str], advanced_feature=None) -> Dict[str, Any]:
        results = []
        for op, expr in zip(operations, expressions):
            # Try symbolic first
            try:
                symbolic_result = self.symbolic_engine.compute(op, expr, variables, features)
                # Ensure result is JSON serializable
                symbolic_result = self._make_serializable(symbolic_result)
                results.append({'operation': op, 'expression': str(expr), 'symbolic_result': symbolic_result})
            except Exception as e:
                # Fallback to numeric
                try:
                    numeric_result = self.numerical_engine.compute(op, expr, variables, features)
                    # Ensure result is JSON serializable
                    numeric_result = self._make_serializable(numeric_result)
                    results.append({'operation': op, 'expression': str(expr), 'numeric_result': numeric_result})
                except Exception as e2:
                    results.append({'operation': op, 'expression': str(expr), 'error': str(e2)})
        return {'results': results}
    
    def _make_serializable(self, obj):
        """Convert any sympy or numpy objects to JSON-serializable format"""
        if isinstance(obj, sp.Basic):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            # For complex objects, try to convert to dict
            try:
                return {k: self._make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            except:
                return str(obj)
        else:
            return obj

class SymbolicEngine:
    def compute(self, operation: str, expr: sp.Expr, variables: Dict[str, Any], features: List[str]) -> Any:
        # General symbolic handling
        if operation == 'integrate':
            # Support multiple integrals
            vars_ = [sp.Symbol(v) for v in variables.keys()] if variables else [sp.Symbol('x')]
            result = sp.integrate(expr, *vars_)
            return result  # Return sympy object, will be converted by _make_serializable
        elif operation in ['derive', 'derivative', 'diff']:
            # Support partial derivatives
            vars_ = [sp.Symbol(v) for v in variables.keys()] if variables else [sp.Symbol('x')]
            return sp.diff(expr, *vars_)
        elif operation == 'gradient':
            vars_ = [sp.Symbol(v) for v in variables.keys()] if variables else [sp.Symbol('x'), sp.Symbol('y')]
            return [sp.diff(expr, v) for v in vars_]
        elif operation == 'jacobian':
            # Assume expr is a list of expressions
            vars_ = [sp.Symbol(v) for v in variables.keys()] if variables else [sp.Symbol('x'), sp.Symbol('y')]
            if isinstance(expr, list):
                return sp.Matrix(expr).jacobian(vars_)
            else:
                return sp.Matrix([expr]).jacobian(vars_)
        elif operation == 'hessian':
            vars_ = [sp.Symbol(v) for v in variables.keys()] if variables else [sp.Symbol('x'), sp.Symbol('y')]
            return sp.hessian(expr, vars_)
        elif operation == 'limit':
            var = list(variables.keys())[0] if variables else 'x'
            point = variables[var] if var in variables else 0
            return sp.limit(expr, sp.Symbol(var), point)
        elif operation == 'solve':
            # Support systems of equations
            vars_ = [sp.Symbol(v) for v in variables.keys()] if variables else [sp.Symbol('x')]
            return sp.solve(expr, *vars_)
        elif operation == 'series':
            var = list(variables.keys())[0] if variables else 'x'
            return sp.series(expr, sp.Symbol(var))
        elif operation == 'determinant':
            return sp.Matrix(expr).det()
        elif operation == 'eigenvalues':
            return sp.Matrix(expr).eigenvals()
        elif operation == 'eigenvectors':
            return sp.Matrix(expr).eigenvects()
        elif operation == 'svd':
            return sp.Matrix(expr).SVD()
        elif operation == 'matrix_inverse':
            return sp.Matrix(expr).inv()
        elif operation == 'matrix_rank':
            return sp.Matrix(expr).rank()
        elif operation == 'mean':
            return sp.stats.E(expr)
        elif operation == 'variance':
            return sp.stats.variance(expr)
        elif operation == 'ode':
            # expr should be an equation, variables should include function and variable
            # Example: expr = sp.Eq(f(x).diff(x), f(x)), variables = {'f': 'f', 'x': 'x'}
            try:
                f = list(variables.keys())[0]
                x = list(variables.keys())[1]
                func = sp.Function(f)
                var = sp.Symbol(x)
                return sp.dsolve(expr, func(var))
            except Exception:
                return "ODE solving failed"
        elif operation == 'pde':
            # Placeholder: SymPy has limited PDE support
            return "PDE solving not fully supported"
        elif operation == 'residue':
            var = list(variables.keys())[0] if variables else 'z'
            point = variables[var] if var in variables else 0
            return sp.residue(expr, sp.Symbol(var), point)
        elif operation == 'combinatorics':
            # Example: expr = sp.binomial(n, k)
            return expr
        elif operation == 'plot':
            # For plot, just return the expression (visualizer will handle)
            return expr
        else:
            # Try to evaluate any other operation symbolically
            try:
                return expr.doit()
            except Exception:
                return expr

class NumericalEngine:
    def compute(self, operation: str, expr: sp.Expr, variables: Dict[str, Any], features: List[str]) -> Any:
        import scipy.stats as stats
        try:
            if operation == 'integrate':
                # Numeric integration (single or multiple)
                if len(variables) == 1:
                    var = list(variables.keys())[0]
                    a, b = variables[var] if isinstance(variables[var], (list, tuple)) else (-10, 10)
                    f = sp.lambdify(var, expr, modules=['numpy'])
                    result, _ = integrate.quad(f, a, b)
                    return result
                elif len(variables) == 2:
                    vars_ = list(variables.keys())
                    a1, b1 = variables[vars_[0]]
                    a2, b2 = variables[vars_[1]]
                    f = sp.lambdify(vars_, expr, modules=['numpy'])
                    result, _ = integrate.dblquad(lambda y, x: f(x, y), a1, b1, lambda x: a2, lambda x: b2)
                    return result
            elif operation in ['derive', 'derivative', 'diff']:
                # Numeric derivative (finite difference)
                var = list(variables.keys())[0] if variables else 'x'
                f = sp.lambdify(var, expr, modules=['numpy'])
                x0 = variables[var] if var in variables else 0
                h = 1e-5
                return (f(x0 + h) - f(x0 - h)) / (2 * h)
            elif operation == 'solve':
                # Numeric root finding
                var = list(variables.keys())[0] if variables else 'x'
                f = sp.lambdify(var, expr, modules=['numpy'])
                x0 = variables[var] if var in variables else 0
                sol = optimize.root_scalar(f, bracket=[x0-10, x0+10], method='brentq')
                return sol.root if sol.converged else "No root found"
            elif operation == 'mean':
                # Numeric mean for a list or array
                if isinstance(expr, (list, np.ndarray)):
                    return np.mean(expr)
                else:
                    f = sp.lambdify(list(variables.keys()), expr, modules=['numpy'])
                    vals = [variables[k] for k in variables]
                    return np.mean(f(*vals))
            elif operation == 'variance':
                if isinstance(expr, (list, np.ndarray)):
                    return np.var(expr)
                else:
                    f = sp.lambdify(list(variables.keys()), expr, modules=['numpy'])
                    vals = [variables[k] for k in variables]
                    return np.var(f(*vals))
            elif operation == 'probability':
                # Example: expr = stats.norm(loc, scale)
                # Notation: variables = {'loc': 0, 'scale': 1, 'x': 1.5}
                dist = getattr(stats, features[0]) if features else stats.norm
                params = {k: v for k, v in variables.items() if k in dist.shapes or k in ['loc', 'scale']}
                x = variables.get('x', 0)
                return dist(**params).cdf(x)
            elif operation == 'matrix_inverse':
                return np.linalg.inv(np.array(expr).astype(float))
            elif operation == 'matrix_rank':
                return np.linalg.matrix_rank(np.array(expr).astype(float))
            elif operation == 'determinant':
                return np.linalg.det(np.array(expr).astype(float))
            elif operation == 'eigenvalues':
                return np.linalg.eigvals(np.array(expr).astype(float))
            elif operation == 'eigenvectors':
                vals, vecs = np.linalg.eig(np.array(expr).astype(float))
                return {'eigenvalues': vals, 'eigenvectors': vecs}
            elif operation == 'svd':
                u, s, vh = np.linalg.svd(np.array(expr).astype(float))
                return {'U': u, 'S': s, 'VH': vh}
            elif operation == 'ode':
                # Numeric ODE solver (simple 1st order)
                from scipy.integrate import solve_ivp
                def fun(t, y):
                    f = sp.lambdify(['t', 'y'], expr, modules=['numpy'])
                    return f(t, y)
                y0 = variables.get('y0', 1)
                t_span = variables.get('t_span', [0, 10])
                sol = solve_ivp(fun, t_span, [y0])
                return {'t': sol.t.tolist(), 'y': sol.y[0].tolist()}
            elif operation == 'pde':
                return "Numeric PDE solving not implemented"
            else:
                # Try to evaluate numerically
                f = sp.lambdify(list(variables.keys()), expr, modules=['numpy'])
                vals = [variables[k] for k in variables]
                return f(*vals)
        except Exception as e:
            return f"Numerical evaluation failed: {str(e)}"
