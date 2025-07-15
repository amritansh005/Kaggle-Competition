from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Dict, Any, Optional
import sqlite3
import os
import random
from datetime import datetime
from backend.mcq_generator import generate_mcqs_for_exam

app = FastAPI(
    title="Personalized Exam Simulator (Multi-Subject)",
    description="Offline, adaptive Olympiad/mock exam generator and evaluator with multi-subject support.",
    version="2.0.0"
)

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

# TopicRequest removed
class SubjectSelection(BaseModel):
    subject: str  # "physics", "chemistry", "biology"
    grade: str    # "11", "12", or "random"
    difficulty: Literal["easy", "medium", "hard"] = "easy"  # Only allow these values

class ExamRequest(BaseModel):
    subjects: list[SubjectSelection]  # List of selected subjects with their options
    language: str = "en"
    user_id: str

class AnswerSubmission(BaseModel):
    user_id: str
    exam_id: str
    answers: dict  # {question_id: answer}
    questions: Optional[list] = None  # MCQ-enriched questions (optional)

class FeedbackRequest(BaseModel):
    user_id: str
    exam_id: str

class MCQQuestionsRequest(BaseModel):
    questions: List[Dict[str, Any]]

# --- In-memory user state (for demo, replace with persistent storage for production) ---
user_progress = {}
exams = {}

# --- Helper functions ---

DB_CONFIG = [
    # (subject, grade, db_path, table_name)
    ("Biology", "11", "NCERT_Biology_11th/Biology_11th_Cleaned.sqlite", "Biology_11th_Cleaned"),
    ("Biology", "12", "NCERT_Biology_12th/Biology_12th_Cleaned.sqlite", "Biology_12th_Cleaned"),
    ("Chemistry", "11", "NCERT_Chemistry_11th/Chemsitry_11th_Cleaned.sqlite", "Chemsitry_11th_Cleaned"),
    ("Chemistry", "12", "NCERT_Chemistry_12th/Chemsitry_12th_Cleaned.sqlite", "Chemsitry_12th_Cleaned"),
    ("Physics", "11", "NCERT_Physics_11th/Physics_11th_Cleaned.sqlite", "Physics_11th_Cleaned"),
    ("Physics", "12", "NCERT_Physics_12th/Physics_12th_Cleaned.sqlite", "Physics_12th_Cleaned"),
]

def get_db_configs(subject, grade):
    # subject/grade can be "random"
    subjects = ["Biology", "Chemistry", "Physics"]
    grades = ["11", "12"]
    selected = []
    if subject == "random":
        subject_choices = [random.choice(subjects)]
    else:
        subject_choices = [subject.capitalize()]
    if grade == "random":
        grade_choices = [random.choice(grades)]
    else:
        grade_choices = [str(grade)]
    for s in subject_choices:
        for g in grade_choices:
            for conf in DB_CONFIG:
                if conf[0] == s and conf[1] == g:
                    selected.append(conf)
    return selected

def fetch_questions_with_filters(subject, grade, difficulty="easy"):
    dbs = get_db_configs(subject, grade)
    all_questions = []
    for subj, grd, db_path, table in dbs:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        # Build query without topic filter
        query = f"SELECT * FROM {table}"
        params = []
        where_clauses = []
        if difficulty:
            where_clauses.append("Difficulty = ?")
            params.append(difficulty.capitalize())
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        for row in rows:
            q = dict(zip(columns, row))
            q["subject"] = subj  # Tag question with subject
            q["grade"] = grd
            all_questions.append(q)
        # DEBUG: Log how many questions were found for this filter
        with open("exam_debug.log", "a", encoding="utf-8") as f:
            f.write(f"DB: {db_path}, Table: {table}, Subject: {subj}, Grade: {grd}, Difficulty: {difficulty}, Found: {len(rows)}\n")
        conn.close()
    if not all_questions:
        raise HTTPException(status_code=404, detail=f"No questions found for {subject} {grade} with the selected filters.")
    # Return all matching questions (no sampling)
    return all_questions

# --- API Endpoints ---

# /get_topics endpoint removed
@app.get("/subjects")
def list_subjects():
    # Hardcoded for now; could scan DBs
    return {
        "subjects": [
            {"subject": "Biology", "grades": ["11th", "12th"]},
            {"subject": "Chemistry", "grades": ["11th", "12th"]},
            {"subject": "Physics", "grades": ["11th", "12th"]},
        ]
    }

@app.post("/generate_exam")
def generate_exam(req: ExamRequest):
    # req.subjects: list of SubjectSelection
    all_questions = []
    filters = []
    for subj_sel in req.subjects:
        # If grade is "random", pick a random grade and use it for this subject
        grade = subj_sel.grade
        if grade == "random":
            grade = random.choice(["11", "12"])
        # Ignore topic, just fetch by subject, grade, difficulty
        questions = fetch_questions_with_filters(
            subj_sel.subject,
            grade,
            subj_sel.difficulty
        )
        # Limit to 5 questions per subject if more are available
        if len(questions) > 5:
            questions = random.sample(questions, 5)
        all_questions.extend(questions)
        filters.append({
            "subject": subj_sel.subject,
            "grade": grade,
            "difficulty": subj_sel.difficulty
        })
    if not all_questions:
        raise HTTPException(status_code=404, detail="No questions found for the selected filters.")

    exam_id = f"{req.user_id}_{random.randint(10000,99999)}"
    # DEBUG: Log the number and subjects of questions being returned
    with open("exam_debug.log", "a", encoding="utf-8") as f:
        f.write(f"Exam ID: {req.user_id}_{exam_id}\n")
        f.write(f"Requested subjects: {filters}\n")
        f.write(f"Number of questions returned: {len(all_questions)}\n")
        f.write(f"Subjects: {[q.get('subject') for q in all_questions]}\n")
        f.write("="*40 + "\n")

    test_obj = {
        "exam_id": exam_id,
        "user_id": req.user_id,
        "filters": filters,
        "questions": all_questions,
        "answers": {},
        "score": None,
        "status": "created",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    exams[exam_id] = test_obj
    user_progress.setdefault(req.user_id, []).append(exam_id)
    return test_obj

@app.post("/generate_mcqs")
def generate_mcqs(req: MCQQuestionsRequest):
    # Wrap in exam-like dict for compatibility with mcq_generator
    exam = {"questions": req.questions}
    mcq_results = generate_mcqs_for_exam(exam)
    return {"mcqs": mcq_results}

@app.post("/submit_answers")
def submit_answers(sub: AnswerSubmission):
    import json
    from datetime import datetime

    exam = exams.get(sub.exam_id)
    if not exam or exam["user_id"] != sub.user_id:
        raise HTTPException(status_code=404, detail="Exam not found for user.")
    exam["answers"] = sub.answers

    # --- Use MCQ-enriched questions if provided ---
    # Accepts: { ... "questions": [...] } in the POST body
    import inspect
    # Try to get MCQ-enriched questions from the request body
    from fastapi import Request
    import sys
    # If using FastAPI v0.95+, you can use sub.dict().get("questions")
    mcq_questions = getattr(sub, "questions", None)
    if mcq_questions is None and hasattr(sub, "__dict__"):
        mcq_questions = sub.__dict__.get("questions")
    if mcq_questions and isinstance(mcq_questions, list):
        exam["questions"] = mcq_questions

    # Auto-evaluate (assume 'answer' column in DB)
    correct = 0
    total = len(exam["questions"])
    for idx, q in enumerate(exam["questions"]):
        qid = str(q.get("id") or q.get("question_id") or q.get("QID") or q.get("qid") or q.get("index") or idx)
        user_ans = sub.answers.get(qid)
        # Use MCQ correct option if available, else fallback to DB answer
        options = q.get("options")
        answer_index = q.get("answer_index")
        if options and answer_index is not None and 0 <= answer_index < len(options):
            correct_ans = str(options[answer_index])
        else:
            correct_ans = str(q.get("answer") or q.get("Answer") or q.get("correct_answer") or "")
        if user_ans is not None and user_ans.strip().lower() == correct_ans.strip().lower():
            correct += 1
    score = correct / total if total else 0
    exam["score"] = score

    # --- Save submission to file ---
    # Build a detailed questions_with_answers array for robust review
    questions_with_answers = []
    for idx, q in enumerate(exam["questions"]):
        qid = str(q.get("id") or q.get("question_id") or q.get("QID") or q.get("qid") or q.get("index") or idx)
        user_ans = sub.answers.get(qid, "")
        options = q.get("options")
        answer_index = q.get("answer_index")
        if options and answer_index is not None and 0 <= answer_index < len(options):
            correct_ans = str(options[answer_index])
        else:
            correct_ans = str(q.get("answer") or q.get("Answer") or q.get("correct_answer") or "")
        questions_with_answers.append({
            "question_id": qid,
            "question": q.get("question") or q.get("Question") or "",
            "options": options if options else None,
            "user_answer": user_ans,
            "correct_answer": correct_ans,
            "subject": q.get("subject", ""),
            "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", "")
        })

    submission_data = {
        "user_id": sub.user_id,
        "exam_id": sub.exam_id,
        "answers": sub.answers,
        "questions": exam["questions"],
        "questions_with_answers": questions_with_answers,
        "score": score,
        "correct": correct,
        "total": total,
        "filters": exam.get("filters", {}),
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    submissions_dir = os.path.join(os.path.dirname(__file__), "submissions")
    os.makedirs(submissions_dir, exist_ok=True)
    filename = f"{sub.exam_id}_{sub.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(submissions_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)

    return {"score": score, "correct": correct, "total": total}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # Stub for Gemma-powered feedback
    exam = exams.get(req.exam_id)
    if not exam or exam["user_id"] != req.user_id:
        raise HTTPException(status_code=404, detail="Exam not found for user.")
    # Placeholder: In real app, use AI to generate feedback
    score = exam.get("score")
    if score is None:
        return {"feedback": "Please submit answers first."}
    if score == 1.0:
        fb = "Excellent! You got all questions correct."
    elif score >= 0.7:
        fb = "Good job! Review the questions you missed for improvement."
    elif score >= 0.4:
        fb = "Keep practicing. Focus on your weak areas."
    else:
        fb = "Don't give up! Try easier questions or review the material."
    return {"feedback": fb}

@app.get("/user_submissions/{user_id}")
def list_user_submissions(user_id: str):
    """
    List all submission files for a given user_id.
    Returns a list of submission metadata (filename, exam_id, submitted_at, score, subject info, etc.).
    """
    import json

    submissions_dir = os.path.join(os.path.dirname(__file__), "submissions")
    if not os.path.exists(submissions_dir):
        return []
    files = [f for f in os.listdir(submissions_dir) if f.endswith(".json") and user_id in f]
    submissions = []
    for fname in sorted(files, reverse=True):
        fpath = os.path.join(submissions_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Only include minimal info for listing
                submissions.append({
                    "filename": fname,
                    "exam_id": data.get("exam_id"),
                    "submitted_at": data.get("submitted_at"),
                    "score": data.get("score"),
                    "filters": data.get("filters"),
                    "total": data.get("total"),
                    "correct": data.get("correct"),
                })
        except Exception as e:
            continue
    # Sort submissions by submitted_at in descending order (most recent first)
    from datetime import datetime
    def parse_dt(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min
    submissions = sorted(submissions, key=lambda x: parse_dt(x.get("submitted_at", "")), reverse=True)
    return submissions

@app.get("/submission/{filename}")
def get_submission(filename: str):
    """
    Fetch a formatted review of a specific submission file.
    Returns: {
        "submitted_at": ...,
        "score": ...,
        "correct": ...,
        "total": ...,
        "filters": ...,
        "questions_with_answers": [
            {
                "question_id": ...,
                "question": ...,
                "options": ...,
                "user_answer": ...,
                "correct_answer": ...,
                "subject": ...,
                "topic": ...,
                "difficulty": ...
            },
            ...
        ]
    }
    """
    import json

    submissions_dir = os.path.join(os.path.dirname(__file__), "submissions")
    fpath = os.path.join(submissions_dir, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Submission file not found.")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Only return the review-relevant fields
    return {
        "submitted_at": data.get("submitted_at"),
        "score": data.get("score"),
        "correct": data.get("correct"),
        "total": data.get("total"),
        "filters": data.get("filters"),
        "questions_with_answers": data.get("questions_with_answers", [])
    }

@app.get("/exam/{exam_id}")
def get_exam(exam_id: str):
    exam = exams.get(exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found.")
    return exam

@app.get("/")
def root():
    return {"message": "Personalized Exam Simulator backend (multi-subject) is running."}

# --- Numerical Solver & Visualizer Endpoint ---
from pydantic import BaseModel
from backend.math_query_processor import MathQueryProcessor
from backend.math_intent_parser import MathIntentParser
from backend.computation_engine import ComputationEngine
from backend.advanced_visualizer import AdvancedVisualizer
from backend.explanation_generator import ExplanationGenerator
from backend.advanced_features import MathVisualizationEngine

class NumericalQueryRequest(BaseModel):
    query: str

import numpy as np
import sympy as sp

def make_json_serializable(obj, visited=None):
    """
    Recursively convert any object to a JSON-serializable format.
    Handles sympy objects, numpy arrays, and other complex types.
    """
    import types
    import math
    import sympy as sp
    import numpy as np
    
    if visited is None:
        visited = set()
    
    # Handle None and basic JSON types first
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
        
    # Handle floats (including special values)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        elif math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    
    # CRITICAL: Handle ALL sympy types before anything else
    # Check for sympy by module name to catch all sympy objects
    if hasattr(obj, '__module__') and obj.__module__ and 'sympy' in obj.__module__:
        return str(obj)
    
    # Also check by class name as a fallback
    if hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
        class_name = obj.__class__.__name__
        if class_name in ['Mul', 'Add', 'Pow', 'Symbol', 'Integer', 'Float', 'Rational']:
            return str(obj)
    
    # Try sympy.Basic check
    try:
        if isinstance(obj, sp.Basic):
            return str(obj)
    except:
        pass
    
    # Handle circular references
    obj_id = id(obj)
    if obj_id in visited:
        return "<circular reference>"
    
    # Only track objects that can be recursive
    if isinstance(obj, (dict, list, tuple)) or hasattr(obj, "__dict__"):
        visited.add(obj_id)
    
    # Handle collections
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v, visited) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item, visited) for item in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(item, visited) for item in obj]
    
    # Handle numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle Plotly figures
    if hasattr(obj, "to_plotly_json"):
        try:
            return obj.to_plotly_json()
        except:
            return {"error": "Failed to convert Plotly figure"}
    
    # Skip properties and descriptors
    if isinstance(obj, (property, types.MemberDescriptorType)):
        return None
    
    # Skip type objects
    if isinstance(obj, type):
        return f"<type {obj.__name__}>"
    
    # Handle objects with __dict__
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            # Check if it's a module
            if hasattr(obj, '__module__') and hasattr(obj, '__name__'):
                return f"<{obj.__module__}.{obj.__name__}>"
            # Try to get a dictionary representation
            obj_dict = {}
            for k, v in obj.__dict__.items():
                if not k.startswith('_'):  # Skip private attributes
                    try:
                        obj_dict[k] = make_json_serializable(v, visited)
                    except:
                        obj_dict[k] = str(v)
            return obj_dict
        except:
            return str(obj)
    
    # Handle pandas if available
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_list()
    except ImportError:
        pass
    
    # Final fallback: convert to string
    try:
        return str(obj)
    except:
        return "<unserializable object>"

from fastapi.responses import JSONResponse
import json

@app.post("/solve_numerical")
def solve_numerical(req: NumericalQueryRequest):
    # Initialize system components
    query_processor = MathQueryProcessor()
    intent_parser = MathIntentParser()
    computation_engine = ComputationEngine()
    visualizer = AdvancedVisualizer()
    advanced_engine = MathVisualizationEngine()
    try:
        import ollama
        llm_client = ollama.Client()
        explanation_generator = ExplanationGenerator(llm_client)
    except Exception:
        explanation_generator = None

    try:
        # 1. Parse natural language query
        parsed_query = query_processor.parse_query(req.query)
        
        # 2. Extract mathematical intent
        math_intent = intent_parser.process(parsed_query)
        
        # 3. Route to appropriate processor
        if math_intent.get('advanced_feature'):
            result = advanced_engine.process_advanced_query(
                math_intent['advanced_feature'],
                {
                    "operations": math_intent.get("operations", []),
                    "expressions": math_intent.get("expressions", []),
                    "variables": math_intent.get("variables", {}),
                    "features": math_intent.get("features", [])
                }
            )
            computation_results = {}
            visualizations = []
            if isinstance(result, dict) and "figure" in result:
                visualizations = [result["figure"]]
            elif isinstance(result, dict) and "visualizations" in result:
                visualizations = result["visualizations"]
            elif hasattr(result, "to_plotly_json"):
                visualizations = [result]
        else:
            computation_results = computation_engine.compute(
                math_intent.get("operations", []),
                math_intent.get("expressions", []),
                math_intent.get("variables", {}),
                math_intent.get("features", [])
            )
            # Deep stringification to guarantee no SymPy objects remain
            def force_stringify(obj):
                try:
                    return str(obj)
                except Exception:
                    return "<unserializable>"
            def deep_stringify(obj):
                if isinstance(obj, dict):
                    return {k: deep_stringify(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_stringify(v) for v in obj]
                else:
                    return force_stringify(obj)
            computation_results = deep_stringify(computation_results)
            visualizations = deep_stringify(
                visualizer.create_visualizations(
                    math_intent,
                    computation_results
                )
            )
            result = {
                'computation': computation_results,
                'visualizations': visualizations
            }
        
        # 4. Generate explanation
        if explanation_generator:
            explanation = explanation_generator.generate_explanation(
                ", ".join(math_intent.get("operations", [])),
                {
                    "expressions": [str(e) for e in math_intent.get("expressions", [])],
                    "variables": math_intent.get("variables", {}),
                    "features": math_intent.get("features", [])
                },
                result,
                {}
            )
            try:
                import markdown
                explanation_html = markdown.markdown(explanation, extensions=['extra', 'sane_lists'])
            except Exception:
                explanation_html = explanation
        else:
            explanation_html = "Explanation service unavailable."

        # Convert Plotly figures to JSON for frontend rendering
        plotly_figs = []
        for viz in visualizations:
            if hasattr(viz, "to_plotly_json"):
                try:
                    fig_json = viz.to_plotly_json()
                    plotly_figs.append(fig_json)
                except Exception as e:
                    print(f"Error converting figure to JSON: {e}")

        # --- PATCH: Ensure frontend-safe structure for computation results ---
        safe_result = result
        # If result is a dict with 'computation', patch its structure
        if isinstance(result, dict) and "computation" in result:
            comp = result["computation"]
            # Ensure comp has 'results' as a list of dicts with 'symbolic_result'
            if not isinstance(comp, dict):
                comp = {}
            results_list = comp.get("results")
            if not (isinstance(results_list, list) and len(results_list) > 0 and "symbolic_result" in results_list[0]):
                # Patch: wrap any available result, or provide a default
                symbolic = ""
                if isinstance(results_list, list) and len(results_list) > 0:
                    symbolic = str(results_list[0])
                elif isinstance(results_list, dict):
                    symbolic = str(results_list)
                elif comp:
                    symbolic = str(comp)
                else:
                    symbolic = "N/A"
                comp["results"] = [{"symbolic_result": symbolic}]
            result["computation"] = comp
            safe_result = result
        # If result is not a dict, wrap it
        elif not isinstance(result, dict):
            safe_result = {
                "computation": {
                    "results": [{"symbolic_result": str(result) if result else "N/A"}]
                },
                "visualizations": []
            }

        # Build response data
        # Use visualizations from results if present, else fallback to plotly_figs
        top_level_visualizations = []
        if isinstance(safe_result, dict) and "visualizations" in safe_result and safe_result["visualizations"]:
            # Make a shallow copy to avoid circular reference
            top_level_visualizations = list(safe_result["visualizations"])
        else:
            top_level_visualizations = plotly_figs

        response_data = {
            "success": True,
            "query": req.query,
            "intent": {
                "operations": math_intent.get("operations", []),
                "expressions": [str(e) for e in math_intent.get("expressions", [])],
                "variables": math_intent.get("variables", {}),
                "features": math_intent.get("features", []),
                "advanced_feature": math_intent.get("advanced_feature")
            },
            "results": safe_result,
            "explanation": explanation_html,
            "visualizations": top_level_visualizations
        }
        
        # Apply make_json_serializable to entire response, but skip "visualizations" field to avoid circular reference
        try:
            # Separate visualizations
            visualizations = response_data.get("visualizations", None)
            response_data_no_viz = dict(response_data)
            if "visualizations" in response_data_no_viz:
                del response_data_no_viz["visualizations"]
            serialized_data = make_json_serializable(response_data_no_viz)
            # Add visualizations back as-is (already JSON-serializable)
            if visualizations is not None:
                serialized_data["visualizations"] = visualizations
        except Exception as e:
            # Fallback: forcibly convert everything to string if serialization fails
            import pprint
            with open("numerical_debug.log", "a", encoding="utf-8") as f:
                f.write("SERIALIZATION ERROR:\n")
                f.write(str(e) + "\n")
                f.write("RAW RESPONSE DATA:\n")
                pprint.pprint(response_data, stream=f)
                f.write("\n" + "="*60 + "\n")
            def force_stringify(obj):
                try:
                    return str(obj)
                except Exception:
                    return "<unserializable>"
            def deep_stringify(obj):
                if isinstance(obj, dict):
                    return {k: deep_stringify(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_stringify(v) for v in obj]
                else:
                    return force_stringify(obj)
            serialized_data = deep_stringify(response_data)
        
        # Debug: log type and content of response
        with open("numerical_debug.log", "a", encoding="utf-8") as f:
            import pprint
            f.write("QUERY: " + req.query + "\n")
            f.write("RESPONSE TYPE: " + str(type(serialized_data)) + "\n")
            f.write("RESPONSE CONTENT:\n")
            pprint.pprint(serialized_data, stream=f)
            # Log the top-level visualizations field
            try:
                f.write("\nTOP-LEVEL VISUALIZATIONS:\n")
                f.write(str(serialized_data.get("visualizations", "NO FIELD")) + "\n")
            except Exception as e:
                f.write(f"\nERROR logging visualizations: {e}\n")
            f.write("\n" + "="*60 + "\n")
        
        # Use JSONResponse with custom encoder to bypass FastAPI's default encoder
        return JSONResponse(content=serialized_data)
        
    except Exception as e:
        # Log the full error for debugging
        import traceback
        with open("numerical_debug.log", "a", encoding="utf-8") as f:
            f.write("ERROR in solve_numerical:\n")
            f.write(f"Query: {req.query}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*60 + "\n")
        
        # Return a safe error response using JSONResponse
        # Ensure error message is always serializable (handles SymPy Mul, etc.)
        try:
            error_msg = str(e)
        except Exception:
            error_msg = repr(e)
        error_response = {
            "success": False,
            "query": req.query,
            "error": error_msg,
            "error_type": type(e).__name__
        }
        return JSONResponse(content=make_json_serializable(error_response))

# --- Chapter-to-Micro-Lesson Compiler Endpoint ---
from pydantic import BaseModel

class MicroLessonRequest(BaseModel):
    chapter_text: str = None
    pdf_base64: str = None

@app.post("/compile_micro_lessons")
def compile_micro_lessons(req: MicroLessonRequest):
    """
    Accepts: {chapter_text: "..."} or {pdf_base64: "..."}
    Returns: {micro_lessons: [...], images: [...]}
    """
    try:
        from backend.computation_engine import (
            extract_text_and_images_from_pdf,
            compile_micro_lessons as compile_micro_lessons_fn,
        )
        chapter_text = req.chapter_text
        page_offsets = []
        if req.pdf_base64:
            # Extract text and page offsets from PDF
            chapter_text, page_offsets = extract_text_and_images_from_pdf(req.pdf_base64)
        if not chapter_text:
            raise HTTPException(status_code=400, detail="No chapter text or PDF provided")
        # Pass page_offsets to compile_micro_lessons for per-section page mapping
        result = compile_micro_lessons_fn(chapter_text, page_offsets)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
