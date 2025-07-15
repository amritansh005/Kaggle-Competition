import plotly.graph_objects as go
import numpy as np
import sympy as sp
from typing import Dict, Any, List

class AdvancedVisualizer:
    def create_visualizations(self, math_intent: Dict[str, Any], computation_results: Dict[str, Any]) -> List[go.Figure]:
        figures = []
        expressions = math_intent.get('expressions', [])
        operations = math_intent.get('operations', [])
        results = computation_results.get('results', [])
        variables = list(math_intent.get('variables', {}).keys())

        for i, expr in enumerate(expressions):
            op = operations[i] if i < len(operations) else "expression"
            # 1D function plot
            if len(variables) == 1:
                var = variables[0]
                x = np.linspace(-10, 10, 1000)
                try:
                    f = sp.lambdify(var, expr, modules=['numpy'])
                    y = f(x)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{op}({var})'))
                    fig.update_layout(
                        title=f"{op.capitalize()} Visualization",
                        xaxis_title=var,
                        yaxis_title=f"{op}({var})",
                        hovermode='x unified'
                    )
                    figures.append(fig)
                except Exception:
                    fig = go.Figure()
                    fig.add_annotation(text=f"{op}: {sp.pretty(expr)}", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
                    fig.update_layout(title=f"{op.capitalize()} (Text)", xaxis_visible=False, yaxis_visible=False)
                    figures.append(fig)
            # 2D implicit plot (e.g., x^2 + y^2 = 1)
            elif len(variables) == 2 and (op == "implicit" or (isinstance(expr, sp.Equality) or (isinstance(expr, sp.Expr) and "=" in str(expr)))):
                var1, var2 = variables[0], variables[1]
                x = np.linspace(-10, 10, 400)
                y = np.linspace(-10, 10, 400)
                X, Y = np.meshgrid(x, y)
                try:
                    if isinstance(expr, sp.Equality):
                        f = sp.lambdify((var1, var2), expr.lhs - expr.rhs, modules=['numpy'])
                    else:
                        # Try to parse as "lhs = rhs"
                        parts = str(expr).split("=")
                        if len(parts) == 2:
                            lhs = sp.sympify(parts[0])
                            rhs = sp.sympify(parts[1])
                            f = sp.lambdify((var1, var2), lhs - rhs, modules=['numpy'])
                        else:
                            f = sp.lambdify((var1, var2), expr, modules=['numpy'])
                    Z = f(X, Y)
                    fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, contours_coloring='lines', showscale=False))
                    fig.update_layout(
                        title=f"Implicit Plot: {sp.pretty(expr)}",
                        xaxis_title=var1,
                        yaxis_title=var2
                    )
                    figures.append(fig)
                except Exception:
                    fig = go.Figure()
                    fig.add_annotation(text=f"Implicit: {sp.pretty(expr)}", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
                    fig.update_layout(title=f"Implicit (Text)", xaxis_visible=False, yaxis_visible=False)
                    figures.append(fig)
            # 3D surface plot (z = f(x, y))
            elif len(variables) == 2 and op in ["plot", "surface", "3d"]:
                var1, var2 = variables[0], variables[1]
                x = np.linspace(-10, 10, 100)
                y = np.linspace(-10, 10, 100)
                X, Y = np.meshgrid(x, y)
                try:
                    f = sp.lambdify((var1, var2), expr, modules=['numpy'])
                    Z = f(X, Y)
                    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
                    fig.update_layout(
                        title=f"3D Surface: {sp.pretty(expr)}",
                        scene=dict(
                            xaxis_title=var1,
                            yaxis_title=var2,
                            zaxis_title="z"
                        )
                    )
                    figures.append(fig)
                except Exception:
                    fig = go.Figure()
                    fig.add_annotation(text=f"3D: {sp.pretty(expr)}", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
                    fig.update_layout(title=f"3D (Text)", xaxis_visible=False, yaxis_visible=False)
                    figures.append(fig)
            # Parametric plot (x(t), y(t), z(t))
            elif op == "parametric":
                try:
                    t = np.linspace(-10, 10, 1000)
                    if len(expr) == 2:
                        f1 = sp.lambdify('t', expr[0], modules=['numpy'])
                        f2 = sp.lambdify('t', expr[1], modules=['numpy'])
                        x = f1(t)
                        y = f2(t)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Parametric Curve'))
                        fig.update_layout(title="Parametric 2D Curve", xaxis_title="x(t)", yaxis_title="y(t)")
                        figures.append(fig)
                    elif len(expr) == 3:
                        f1 = sp.lambdify('t', expr[0], modules=['numpy'])
                        f2 = sp.lambdify('t', expr[1], modules=['numpy'])
                        f3 = sp.lambdify('t', expr[2], modules=['numpy'])
                        x = f1(t)
                        y = f2(t)
                        z = f3(t)
                        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Parametric 3D Curve')])
                        fig.update_layout(title="Parametric 3D Curve")
                        figures.append(fig)
                except Exception:
                    fig = go.Figure()
                    fig.add_annotation(text=f"Parametric: {sp.pretty(expr)}", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
                    fig.update_layout(title=f"Parametric (Text)", xaxis_visible=False, yaxis_visible=False)
                    figures.append(fig)
            # Vector field plot (2D)
            elif op in ["vector_field", "field"] and len(variables) == 2:
                var1, var2 = variables[0], variables[1]
                x = np.linspace(-5, 5, 20)
                y = np.linspace(-5, 5, 20)
                X, Y = np.meshgrid(x, y)
                try:
                    # expr should be a tuple/list: (fx, fy)
                    fx, fy = expr
                    fx_func = sp.lambdify((var1, var2), fx, modules=['numpy'])
                    fy_func = sp.lambdify((var1, var2), fy, modules=['numpy'])
                    U = fx_func(X, Y)
                    V = fy_func(X, Y)
                    fig = go.Figure(data=go.Cone(x=X.flatten(), y=Y.flatten(), z=np.zeros_like(X.flatten()),
                                                 u=U.flatten(), v=V.flatten(), w=np.zeros_like(U.flatten()),
                                                 colorscale='Viridis', sizemode="absolute", sizeref=2))
                    fig.update_layout(title="2D Vector Field", scene=dict(
                        xaxis_title=var1, yaxis_title=var2, zaxis_title=""
                    ))
                    figures.append(fig)
                except Exception:
                    fig = go.Figure()
                    fig.add_annotation(text=f"Vector Field: {sp.pretty(expr)}", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
                    fig.update_layout(title=f"Vector Field (Text)", xaxis_visible=False, yaxis_visible=False)
                    figures.append(fig)
            else:
                # Fallback: show as text
                fig = go.Figure()
                fig.add_annotation(text=f"{op}: {sp.pretty(expr)}", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
                fig.update_layout(title=f"{op.capitalize()} (Text)", xaxis_visible=False, yaxis_visible=False)
                figures.append(fig)

        # Also visualize results if present and not already visualized
        for res in results:
            if 'symbolic_result' in res:
                val = res['symbolic_result']
            elif 'numeric_result' in res:
                val = res['numeric_result']
            else:
                val = res.get('error', 'No result')
            fig = go.Figure()
            fig.add_annotation(text=f"Result: {val}", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig.update_layout(title="Computation Result", xaxis_visible=False, yaxis_visible=False)
            figures.append(fig)

        # Always return JSON-serializable dicts for all figures
        return [fig.to_plotly_json() for fig in figures]
