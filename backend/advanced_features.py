# advanced_features.py - Implementation of Advanced Mathematical Features

import numpy as np
import sympy as sp
from scipy import integrate, optimize, signal, stats
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from statsmodels import api as sm
from typing import Dict, List, Tuple, Any, Callable
import pandas as pd

class ComplexAnalysisVisualizer:
    """Handles complex function visualizations"""
    
    def plot_complex_function(self, func: Callable, 
                            x_range: Tuple[float, float] = (-2, 2),
                            y_range: Tuple[float, float] = (-2, 2),
                            resolution: int = 400) -> go.Figure:
        """
        Visualize complex function using domain coloring.
        Magnitude shown as brightness, phase as hue.
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        # Evaluate complex function
        W = np.vectorize(func)(Z)
        
        # Extract magnitude and phase
        magnitude = np.abs(W)
        phase = np.angle(W)
        
        # Create HSV representation
        hue = (phase + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        saturation = np.ones_like(hue)
        value = 1 - 1 / (1 + magnitude**0.3)  # Compress magnitude
        
        # Convert HSV to RGB
        rgb_image = self._hsv_to_rgb(hue, saturation, value)
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap for magnitude
        fig.add_trace(go.Heatmap(
            z=magnitude,
            x=x,
            y=y,
            colorscale='Viridis',
            name='Magnitude',
            visible=True,
            colorbar=dict(title='|f(z)|', x=1.02)
        ))
        
        # Add contour for phase
        fig.add_trace(go.Contour(
            z=phase,
            x=x,
            y=y,
            colorscale='HSV',
            name='Phase',
            visible=False,
            colorbar=dict(title='arg(f(z))', x=1.02)
        ))
        
        # Add buttons for switching views
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label="Magnitude",
                             method="update",
                             args=[{"visible": [True, False]}]),
                        dict(label="Phase",
                             method="update",
                             args=[{"visible": [False, True]}]),
                        dict(label="Both",
                             method="update",
                             args=[{"visible": [True, True]}])
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.15
                )
            ],
            title="Complex Function Visualization",
            xaxis_title="Real",
            yaxis_title="Imaginary",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_riemann_surface(self, func: Callable, 
                           branch_points: List[complex] = None) -> go.Figure:
        """Create 3D visualization of Riemann surface"""
        # Generate parameter space
        r = np.linspace(0.1, 3, 50)
        theta = np.linspace(0, 4*np.pi, 200)  # Multiple sheets
        R, Theta = np.meshgrid(r, theta)
        
        # Convert to complex plane
        Z = R * np.exp(1j * Theta)
        W = np.vectorize(func)(Z)
        
        # Extract real and imaginary parts for 3D plot
        X = np.real(Z)
        Y = np.imag(Z)
        Z_real = np.real(W)
        Z_imag = np.imag(W)
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=['Real Part', 'Imaginary Part']
        )
        
        # Real part surface
        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z_real, colorscale='Viridis'),
            row=1, col=1
        )
        
        # Imaginary part surface
        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z_imag, colorscale='Plasma'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Riemann Surface Visualization",
            scene=dict(
                xaxis_title="Re(z)",
                yaxis_title="Im(z)",
                zaxis_title="Re(f(z))"
            ),
            scene2=dict(
                xaxis_title="Re(z)",
                yaxis_title="Im(z)",
                zaxis_title="Im(f(z))"
            )
        )
        
        return fig
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB for complex domain coloring"""
        h_i = np.floor(h * 6).astype(int)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        h_i = h_i % 6
        
        conditions = [
            (h_i == 0), (h_i == 1), (h_i == 2),
            (h_i == 3), (h_i == 4), (h_i == 5)
        ]
        
        r_choices = [v, q, p, p, t, v]
        g_choices = [t, v, v, q, p, p]
        b_choices = [p, p, t, v, v, q]
        
        r = np.select(conditions, r_choices)
        g = np.select(conditions, g_choices)
        b = np.select(conditions, b_choices)
        
        return np.stack([r, g, b], axis=-1)


class DifferentialEquationsSolver:
    """Solves and visualizes differential equations"""
    
    def solve_and_visualize_ode(self, 
                               equation_func: Callable,
                               initial_conditions: List[float],
                               t_span: Tuple[float, float],
                               params: Dict = None) -> Dict[str, Any]:
        """
        Solve ODE and create various visualizations.
        
        Args:
            equation_func: Function f(y, t, *params) for dy/dt = f(y, t)
            initial_conditions: Initial values [y0, y'0, ...]
            t_span: Time interval (t0, tf)
            params: Additional parameters for the equation
        """
        # Time points
        t = np.linspace(t_span[0], t_span[1], 1000)
        
        # Solve ODE
        if params:
            solution = odeint(equation_func, initial_conditions, t, args=tuple(params.values()))
        else:
            solution = odeint(equation_func, initial_conditions, t)
        
        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Solution', 'Phase Portrait', 'Direction Field', '3D Phase Space']
        )
        
        # 1. Solution plot
        for i, sol in enumerate(solution.T):
            fig.add_trace(
                go.Scatter(x=t, y=sol, name=f'y_{i}'),
                row=1, col=1
            )
        
        # 2. Phase portrait (for 2D systems)
        if len(initial_conditions) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=solution[:, 0], 
                    y=solution[:, 1],
                    mode='lines',
                    name='Trajectory'
                ),
                row=1, col=2
            )
            
            # Add direction field
            self._add_direction_field(fig, equation_func, solution, row=2, col=1)
        
        # 3. 3D phase space (for 3D systems)
        if len(initial_conditions) >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=solution[:, 0],
                    y=solution[:, 1],
                    z=solution[:, 2],
                    mode='lines',
                    name='3D Trajectory'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Differential Equation Solution and Analysis",
            height=800,
            showlegend=True
        )
        
        return {
            'figure': fig,
            'solution': solution,
            'time': t,
            'stability': self._analyze_stability(equation_func, initial_conditions)
        }
    
    def solve_pde(self, 
                  pde_type: str,
                  boundary_conditions: Dict,
                  domain: Dict,
                  params: Dict = None) -> go.Figure:
        """
        Solve common PDEs (heat, wave, Laplace).
        """
        if pde_type == 'heat':
            return self._solve_heat_equation(boundary_conditions, domain, params)
        elif pde_type == 'wave':
            return self._solve_wave_equation(boundary_conditions, domain, params)
        elif pde_type == 'laplace':
            return self._solve_laplace_equation(boundary_conditions, domain)
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")
    
    def _solve_heat_equation(self, bc: Dict, domain: Dict, params: Dict) -> go.Figure:
        """Solve 1D heat equation using finite differences"""
        # Grid
        nx = domain.get('nx', 100)
        nt = domain.get('nt', 500)
        L = domain.get('L', 1.0)
        T = domain.get('T', 1.0)
        
        dx = L / (nx - 1)
        dt = T / (nt - 1)
        alpha = params.get('alpha', 0.01)  # Thermal diffusivity
        
        # Stability check
        r = alpha * dt / (dx**2)
        if r > 0.5:
            dt = 0.5 * dx**2 / alpha
            nt = int(T / dt) + 1
        
        # Initialize
        u = np.zeros((nt, nx))
        x = np.linspace(0, L, nx)
        t = np.linspace(0, T, nt)
        
        # Initial condition
        u[0, :] = bc['initial'](x)
        
        # Boundary conditions
        u[:, 0] = bc.get('left', 0)
        u[:, -1] = bc.get('right', 0)
        
        # Time stepping
        for n in range(nt - 1):
            for i in range(1, nx - 1):
                u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
        
        # Create animation
        fig = go.Figure()
        
        # Add traces
        for i in range(0, nt, nt // 20):  # Show 20 time steps
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=u[i, :],
                    mode='lines',
                    name=f't = {t[i]:.2f}',
                    visible=False
                )
            )
        
        # Make first trace visible
        fig.data[0].visible = True
        
        # Create slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],
                label=f'{i * T / 20:.2f}'
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        
        sliders = [dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(prefix="Time: ", suffix=" s"),
            steps=steps
        )]
        
        fig.update_layout(
            sliders=sliders,
            title="Heat Equation Solution",
            xaxis_title="Position",
            yaxis_title="Temperature"
        )
        
        return fig
    
    def _add_direction_field(self, fig, equation_func, solution, row, col):
        """Add direction field to phase portrait"""
        # Get bounds from solution
        x_min, x_max = solution[:, 0].min(), solution[:, 0].max()
        y_min, y_max = solution[:, 1].min(), solution[:, 1].max()
        
        # Create grid
        x = np.linspace(x_min - 0.5, x_max + 0.5, 20)
        y = np.linspace(y_min - 0.5, y_max + 0.5, 20)
        X, Y = np.meshgrid(x, y)
        
        # Calculate direction vectors
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                dydt = equation_func([X[i, j], Y[i, j]], 0)
                U[i, j] = dydt[0]
                V[i, j] = dydt[1]
        
        # Normalize
        N = np.sqrt(U**2 + V**2)
        U, V = U / N, V / N
        
        # Add quiver plot
        fig.add_trace(
            go.Scatter(
                x=X.flatten()[::2],
                y=Y.flatten()[::2],
                mode='markers',
                marker=dict(size=1),
                showlegend=False
            ),
            row=row, col=col
        )
    
    def _analyze_stability(self, equation_func, equilibrium_point):
        """Analyze stability of equilibrium point"""
        # This would implement linearization and eigenvalue analysis
        # For now, return placeholder
        return {
            'stable': True,
            'eigenvalues': [],
            'type': 'node'
        }


class StatisticalAnalysisModule:
    """Comprehensive statistical analysis and visualization"""
    
    def statistical_analysis(self, 
                           data: np.ndarray,
                           test_type: str,
                           params: Dict = None) -> Dict[str, Any]:
        """
        Perform statistical analysis with visualizations.
        
        Args:
            data: Input data (can be 1D or 2D array)
            test_type: Type of test ('normality', 't-test', 'anova', 'regression', etc.)
            params: Additional parameters for the test
        """
        results = {}
        
        if test_type == 'normality':
            results = self._normality_test(data)
        elif test_type == 't-test':
            results = self._t_test(data, params)
        elif test_type == 'anova':
            results = self._anova(data, params)
        elif test_type == 'regression':
            results = self._regression_analysis(data, params)
        elif test_type == 'time_series':
            results = self._time_series_analysis(data, params)
        
        return results
    
    def _normality_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Test for normality with multiple methods"""
        from scipy import stats
        
        # Flatten if multidimensional
        data = data.flatten()
        
        # Perform tests
        shapiro_stat, shapiro_p = stats.shapiro(data)
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Histogram with Normal Fit', 'Q-Q Plot', 
                          'Box Plot', 'Density Plot']
        )
        
        # 1. Histogram with normal fit
        hist_data = go.Histogram(
            x=data,
            nbinsx=30,
            name='Data',
            histnorm='probability density'
        )
        
        # Normal distribution overlay
        x_range = np.linspace(data.min(), data.max(), 100)
        normal_fit = stats.norm.pdf(x_range, data.mean(), data.std())
        
        fig.add_trace(hist_data, row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x_range, y=normal_fit, name='Normal Fit'),
            row=1, col=1
        )
        
        # 2. Q-Q Plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        sample_quantiles = np.sort(data)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot'
            ),
            row=1, col=2
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                mode='lines',
                name='Reference',
                line=dict(dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. Box Plot
        fig.add_trace(
            go.Box(y=data, name='Data'),
            row=2, col=1
        )
        
        # 4. Density Plot (KDE)
        kde = stats.gaussian_kde(data)
        density_x = np.linspace(data.min(), data.max(), 200)
        density_y = kde(density_x)
        
        fig.add_trace(
            go.Scatter(x=density_x, y=density_y, name='KDE'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Normality Test Results',
            height=800,
            showlegend=False
        )
        
        return {
            'figure': fig,
            'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p},
            'summary_stats': {
                'mean': data.mean(),
                'std': data.std(),
                'skew': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        }
    
    def _regression_analysis(self, data: Dict, params: Dict) -> Dict[str, Any]:
        """Perform regression analysis with diagnostics"""
        # Extract data
        X = data.get('X')
        y = data.get('y')
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Predictions
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Create diagnostic plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Fitted vs Actual', 'Residual Plot',
                          'Q-Q Plot of Residuals', 'Scale-Location Plot']
        )
        
        # 1. Fitted vs Actual
        fig.add_trace(
            go.Scatter(x=predictions, y=y, mode='markers', name='Data'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[predictions.min(), predictions.max()],
                y=[predictions.min(), predictions.max()],
                mode='lines',
                name='Perfect Fit',
                line=dict(dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Residual Plot
        fig.add_trace(
            go.Scatter(x=predictions, y=residuals, mode='markers'),
            row=1, col=2
        )
        fig.add_shape(
            type='line',
            x0=predictions.min(), x1=predictions.max(),
            y0=0, y1=0,
            line=dict(dash='dash'),
            row=1, col=2
        )
        
        # 3. Q-Q Plot of Residuals
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers'),
            row=2, col=1
        )
        
        # 4. Scale-Location Plot
        standardized_residuals = residuals / np.std(residuals)
        fig.add_trace(
            go.Scatter(
                x=predictions,
                y=np.sqrt(np.abs(standardized_residuals)),
                mode='markers'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Regression Diagnostics',
            height=800,
            showlegend=False
        )
        
        return {
            'figure': fig,
            'model_summary': model.summary(),
            'coefficients': model.params,
            'r_squared': model.rsquared,
            'p_values': model.pvalues,
            'confidence_intervals': model.conf_int()
        }


class LinearAlgebraVisualizer:
    """Visualize linear algebra concepts"""
    
    def visualize_transformation(self, 
                               matrix: np.ndarray,
                               vectors: List[np.ndarray] = None) -> go.Figure:
        """Visualize linear transformation effect on vectors/shapes"""
        if matrix.shape[0] == 2 and matrix.shape[1] == 2:
            return self._visualize_2d_transformation(matrix, vectors)
        elif matrix.shape[0] == 3 and matrix.shape[1] == 3:
            return self._visualize_3d_transformation(matrix, vectors)
        else:
            raise ValueError("Only 2x2 and 3x3 matrices supported for visualization")
    
    def _visualize_2d_transformation(self, matrix: np.ndarray, 
                                    vectors: List[np.ndarray] = None) -> go.Figure:
        """2D transformation visualization"""
        fig = go.Figure()
        
        # Default shapes to transform
        if vectors is None:
            # Unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.array([np.cos(theta), np.sin(theta)])
            
            # Unit square
            square = np.array([
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0]
            ])
            
            # Grid lines
            grid_x = []
            grid_y = []
            for i in np.linspace(-2, 2, 9):
                grid_x.extend([i, i, None])
                grid_y.extend([-2, 2, None])
                grid_x.extend([-2, 2, None])
                grid_y.extend([i, i, None])
            
            grid = np.array([grid_x, grid_y])
            
            shapes = {'circle': circle, 'square': square, 'grid': grid}
        else:
            shapes = {'vectors': np.array(vectors).T}
        
        # Apply transformation
        for name, shape in shapes.items():
            # Original shape
            fig.add_trace(go.Scatter(
                x=shape[0],
                y=shape[1],
                mode='lines' if name != 'vectors' else 'lines+markers',
                name=f'Original {name}',
                line=dict(dash='dot'),
                opacity=0.5
            ))
            
            # Transformed shape
            transformed = matrix @ shape
            fig.add_trace(go.Scatter(
                x=transformed[0],
                y=transformed[1],
                mode='lines' if name != 'vectors' else 'lines+markers',
                name=f'Transformed {name}'
            ))
        
        # Add eigenvectors if they exist
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
            if np.isreal(eigenvalue):
                # Original eigenvector
                fig.add_trace(go.Scatter(
                    x=[0, eigenvector[0]],
                    y=[0, eigenvector[1]],
                    mode='lines+markers',
                    name=f'Eigenvector {i+1} (λ={eigenvalue:.2f})',
                    line=dict(width=3)
                ))
        
        fig.update_layout(
            title=f'Linear Transformation: det={np.linalg.det(matrix):.2f}',
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(scaleanchor='x', scaleratio=1),
            showlegend=True
        )
        
        return fig
    
    def visualize_eigenvalues(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Visualize eigenvalues and eigenvectors"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Create visualizations based on matrix size
        if matrix.shape[0] == 2:
            fig = self._visualize_2d_eigen(matrix, eigenvalues, eigenvectors)
        else:
            fig = self._visualize_general_eigen(eigenvalues, eigenvectors)
        
        return {
            'figure': fig,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'characteristic_polynomial': self._get_characteristic_poly(matrix)
        }
    
    def _get_characteristic_poly(self, matrix: np.ndarray):
        """Get characteristic polynomial"""
        n = matrix.shape[0]
        x = sp.Symbol('x')
        I = sp.eye(n)
        char_matrix = sp.Matrix(matrix) - x * I
        char_poly = char_matrix.det()
        return str(char_poly)


class VectorFieldVisualizer:
    """Visualize vector fields and flow"""
    
    def plot_vector_field(self, 
                         field_func: Callable,
                         x_range: Tuple[float, float],
                         y_range: Tuple[float, float],
                         density: int = 20) -> go.Figure:
        """Plot 2D vector field with streamlines"""
        # Create grid
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                U[i, j], V[i, j] = field_func(X[i, j], Y[i, j])
        
        # Create figure
        fig = go.Figure()
        
        # Add quiver plot
        fig.add_trace(go.Cone(
            x=X.flatten(),
            y=Y.flatten(),
            z=np.zeros(X.size),
            u=U.flatten(),
            v=V.flatten(),
            w=np.zeros(X.size),
            sizemode='scaled',
            sizeref=0.5,
            anchor='tail',
            showscale=False,
            colorscale='Blues',
            name='Vector Field'
        ))
        
        # Add streamlines
        fig.add_trace(go.Streamline(
            x=x,
            y=y,
            u=U.T,
            v=V.T,
            name='Streamlines'
        ))
        
        # Calculate and plot critical points
        critical_points = self._find_critical_points(field_func, x_range, y_range)
        if critical_points:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in critical_points],
                y=[p[1] for p in critical_points],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Critical Points'
            ))
        
        fig.update_layout(
            title='Vector Field Visualization',
            xaxis_title='x',
            yaxis_title='y',
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def plot_3d_vector_field(self,
                           field_func: Callable,
                           x_range: Tuple[float, float],
                           y_range: Tuple[float, float],
                           z_range: Tuple[float, float],
                           density: int = 10) -> go.Figure:
        """Plot 3D vector field"""
        # Create 3D grid
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        z = np.linspace(z_range[0], z_range[1], density)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Evaluate vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        W = np.zeros_like(Z)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    U[i,j,k], V[i,j,k], W[i,j,k] = field_func(X[i,j,k], Y[i,j,k], Z[i,j,k])
        
        # Create figure
        fig = go.Figure()
        
        # Add 3D cone plot
        fig.add_trace(go.Cone(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            u=U.flatten(),
            v=V.flatten(),
            w=W.flatten(),
            sizemode='scaled',
            sizeref=0.3,
            anchor='tail',
            colorscale='Viridis',
            name='3D Vector Field'
        ))
        
        fig.update_layout(
            title='3D Vector Field Visualization',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def _find_critical_points(self, field_func, x_range, y_range):
        """Find critical points where field is zero"""
        from scipy.optimize import fsolve
        
        critical_points = []
        
        # Define system of equations
        def system(point):
            return field_func(point[0], point[1])
        
        # Try multiple initial guesses
        for x0 in np.linspace(x_range[0], x_range[1], 5):
            for y0 in np.linspace(y_range[0], y_range[1], 5):
                try:
                    solution = fsolve(system, [x0, y0])
                    # Check if solution is within bounds and unique
                    if (x_range[0] <= solution[0] <= x_range[1] and
                        y_range[0] <= solution[1] <= y_range[1]):
                        # Check if this point is already found
                        is_new = True
                        for cp in critical_points:
                            if np.allclose(solution, cp, atol=1e-6):
                                is_new = False
                                break
                        if is_new:
                            critical_points.append(solution)
                except:
                    pass
        
        return critical_points


class ParametricPlotter:
    """Handle parametric equations and curves"""
    
    def plot_parametric_2d(self,
                          x_func: Callable,
                          y_func: Callable,
                          t_range: Tuple[float, float],
                          num_points: int = 1000) -> go.Figure:
        """Plot 2D parametric curve with velocity and acceleration vectors"""
        t = np.linspace(t_range[0], t_range[1], num_points)
        
        # Calculate position
        x = np.array([x_func(ti) for ti in t])
        y = np.array([y_func(ti) for ti in t])
        
        # Calculate velocity (first derivative)
        dt = t[1] - t[0]
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        
        # Calculate acceleration (second derivative)
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        
        # Calculate speed and curvature
        speed = np.sqrt(vx**2 + vy**2)
        curvature = np.abs(vx * ay - vy * ax) / (speed**3 + 1e-10)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Parametric Curve', 'Speed vs Time',
                          'Curvature', 'Velocity Field']
        )
        
        # 1. Main parametric curve
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name='Curve',
                      line=dict(color=t, colorscale='Viridis', width=3)),
            row=1, col=1
        )
        
        # Add start and end points
        fig.add_trace(
            go.Scatter(x=[x[0]], y=[y[0]], mode='markers',
                      marker=dict(size=10, color='green'),
                      name='Start'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[x[-1]], y=[y[-1]], mode='markers',
                      marker=dict(size=10, color='red'),
                      name='End'),
            row=1, col=1
        )
        
        # 2. Speed plot
        fig.add_trace(
            go.Scatter(x=t, y=speed, mode='lines', name='Speed'),
            row=1, col=2
        )
        
        # 3. Curvature plot
        fig.add_trace(
            go.Scatter(x=t, y=curvature, mode='lines', name='Curvature'),
            row=2, col=1
        )
        
        # 4. Velocity vectors along curve
        sample_indices = np.linspace(0, len(t)-1, 20, dtype=int)
        for i in sample_indices:
            fig.add_trace(
                go.Scatter(
                    x=[x[i], x[i] + 0.1*vx[i]/speed[i]],
                    y=[y[i], y[i] + 0.1*vy[i]/speed[i]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', opacity=0.3,
                      showlegend=False),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="t", row=1, col=2)
        fig.update_yaxes(title_text="Speed", row=1, col=2)
        fig.update_xaxes(title_text="t", row=2, col=1)
        fig.update_yaxes(title_text="Curvature", row=2, col=1)
        
        fig.update_layout(
            title='Parametric Curve Analysis',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_parametric_3d(self,
                          x_func: Callable,
                          y_func: Callable,
                          z_func: Callable,
                          t_range: Tuple[float, float],
                          num_points: int = 1000) -> go.Figure:
        """Plot 3D parametric curve with analysis"""
        t = np.linspace(t_range[0], t_range[1], num_points)
        
        # Calculate position
        x = np.array([x_func(ti) for ti in t])
        y = np.array([y_func(ti) for ti in t])
        z = np.array([z_func(ti) for ti in t])
        
        # Create figure
        fig = go.Figure()
        
        # Add parametric curve
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                color=t,
                colorscale='Viridis',
                width=4,
                colorbar=dict(title='Parameter t')
            ),
            name='3D Curve'
        ))
        
        # Add start and end points
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='End'
        ))
        
        fig.update_layout(
            title='3D Parametric Curve',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            width=800,
            height=600
        )
        
        return fig


class ImplicitFunctionPlotter:
    """Plot implicit functions F(x,y) = 0"""
    
    def plot_implicit_2d(self,
                        func: Callable,
                        x_range: Tuple[float, float],
                        y_range: Tuple[float, float],
                        resolution: int = 200) -> go.Figure:
        """Plot implicit function using contour method"""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(X[i, j], Y[i, j])
        
        # Create figure
        fig = go.Figure()
        
        # Add contour at level 0
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            contours=dict(
                start=0,
                end=0,
                size=0.1,
                coloring='none',
                showlines=True,
                showlabels=False
            ),
            line=dict(width=3),
            name='F(x,y) = 0'
        ))
        
        # Add gradient field
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Calculate gradient
        Fx = np.gradient(Z, dx, axis=1)
        Fy = np.gradient(Z, dy, axis=0)
        
        # Sample points for gradient
        skip = resolution // 20
        x_grad = X[::skip, ::skip].flatten()
        y_grad = Y[::skip, ::skip].flatten()
        fx_grad = Fx[::skip, ::skip].flatten()
        fy_grad = Fy[::skip, ::skip].flatten()
        
        # Normalize gradient
        norm = np.sqrt(fx_grad**2 + fy_grad**2) + 1e-10
        fx_grad /= norm
        fy_grad /= norm
        
        # Add gradient arrows
        for i in range(len(x_grad)):
            fig.add_annotation(
                x=x_grad[i],
                y=y_grad[i],
                ax=x_grad[i] + 0.1*fx_grad[i],
                ay=y_grad[i] + 0.1*fy_grad[i],
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='blue',
                opacity=0.5
            )
        
        fig.update_layout(
            title='Implicit Function Plot with Gradient Field',
            xaxis_title='x',
            yaxis_title='y',
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(scaleanchor='x', scaleratio=1),
            showlegend=True
        )
        
        return fig
    
    def plot_implicit_3d(self,
                        func: Callable,
                        x_range: Tuple[float, float],
                        y_range: Tuple[float, float],
                        z_range: Tuple[float, float],
                        resolution: int = 50) -> go.Figure:
        """Plot implicit surface F(x,y,z) = 0"""
        # Create 3D grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        z = np.linspace(z_range[0], z_range[1], resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Evaluate function
        values = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    values[i, j, k] = func(X[i, j, k], Y[i, j, k], Z[i, j, k])
        
        # Use marching cubes to find isosurface
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(values, 0)
        
        # Scale vertices to actual coordinates
        verts[:, 0] = verts[:, 0] * (x_range[1] - x_range[0]) / resolution + x_range[0]
        verts[:, 1] = verts[:, 1] * (y_range[1] - y_range[0]) / resolution + y_range[0]
        verts[:, 2] = verts[:, 2] * (z_range[1] - z_range[0]) / resolution + z_range[0]
        
        # Create mesh
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.7,
                colorscale='Viridis',
                intensity=verts[:, 2],
                name='F(x,y,z) = 0'
            )
        ])
        
        fig.update_layout(
            title='3D Implicit Surface',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            width=800,
            height=600
        )
        
        return fig


# Integration class to combine all features
class MathVisualizationEngine:
    """Main engine that integrates all mathematical visualization features"""
    
    def __init__(self):
        self.complex_viz = ComplexAnalysisVisualizer()
        self.de_solver = DifferentialEquationsSolver()
        self.stats_module = StatisticalAnalysisModule()
        self.linalg_viz = LinearAlgebraVisualizer()
        self.vector_viz = VectorFieldVisualizer()
        self.param_plotter = ParametricPlotter()
        self.implicit_plotter = ImplicitFunctionPlotter()
    
    def process_advanced_query(self, query_type: str, params: Dict) -> Dict[str, Any]:
        """Route query to appropriate advanced feature"""
        if query_type == 'complex_function':
            return {
                'type': 'complex',
                'result': self.complex_viz.plot_complex_function(
                    params['function'],
                    params.get('x_range', (-2, 2)),
                    params.get('y_range', (-2, 2))
                )
            }
        elif query_type == 'differential_equation':
            return {
                'type': 'de',
                'result': self.de_solver.solve_and_visualize_ode(
                    params['equation'],
                    params['initial_conditions'],
                    params['t_span'],
                    params.get('parameters')
                )
            }
        elif query_type == 'statistical_test':
            return {
                'type': 'stats',
                'result': self.stats_module.statistical_analysis(
                    params['data'],
                    params['test_type'],
                    params.get('test_params')
                )
            }
        elif query_type == 'linear_transformation':
            return {
                'type': 'linalg',
                'result': self.linalg_viz.visualize_transformation(
                    params['matrix'],
                    params.get('vectors')
                )
            }
        elif query_type == 'vector_field':
            return {
                'type': 'vector',
                'result': self.vector_viz.plot_vector_field(
                    params['field_function'],
                    params['x_range'],
                    params['y_range']
                )
            }
        elif query_type == 'parametric':
            if 'z_func' in params:
                return {
                    'type': 'parametric_3d',
                    'result': self.param_plotter.plot_parametric_3d(
                        params['x_func'],
                        params['y_func'],
                        params['z_func'],
                        params['t_range']
                    )
                }
            else:
                return {
                    'type': 'parametric_2d',
                    'result': self.param_plotter.plot_parametric_2d(
                        params['x_func'],
                        params['y_func'],
                        params['t_range']
                    )
                }
        elif query_type == 'implicit':
            if 'z_range' in params:
                return {
                    'type': 'implicit_3d',
                    'result': self.implicit_plotter.plot_implicit_3d(
                        params['function'],
                        params['x_range'],
                        params['y_range'],
                        params['z_range']
                    )
                }
            else:
                return {
                    'type': 'implicit_2d',
                    'result': self.implicit_plotter.plot_implicit_2d(
                        params['function'],
                        params['x_range'],
                        params['y_range']
                    )
                }