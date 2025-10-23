# core/fvg/fvg_visualizer.py
"""
FVG Visualizer - Visualization for Fair Value Gaps

Tasks:
- Draw OHLC chart with FVG zones
- Highlight active vs touched FVGs
- Visualize signals & trades
- Export HTML interactive chart
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict
from datetime import datetime
import os

from .fvg_model import FVG


class FVGVisualizer:
    """
    FVG Visualizer - Create visualizations for FVG analysis

    Attributes:
        show_touched_fvgs: Show touched FVGs (default True)
        show_labels: Show labels on FVG zones (default True)
    """

    def __init__(self, show_touched_fvgs: bool = True, show_labels: bool = True):
        """
        Initialize FVG Visualizer

        Args:
            show_touched_fvgs: Whether to show touched FVGs
            show_labels: Whether to show labels
        """
        self.show_touched_fvgs = show_touched_fvgs
        self.show_labels = show_labels

        # Color schemes
        self.colors = {
            'bullish_active': 'rgba(0, 255, 0, 0.2)',  # Green transparent
            'bearish_active': 'rgba(255, 0, 0, 0.2)',  # Red transparent
            'bullish_touched': 'rgba(128, 128, 128, 0.15)',  # Gray transparent
            'bearish_touched': 'rgba(128, 128, 128, 0.15)',  # Gray transparent
            'bullish_border': 'rgba(0, 200, 0, 0.8)',
            'bearish_border': 'rgba(200, 0, 0, 0.8)',
            'touched_border': 'rgba(100, 100, 100, 0.5)'
        }

    def plot_fvg_chart(self, data: pd.DataFrame, fvgs: List[FVG],
                      title: str = "FVG Analysis Chart",
                      show_volume: bool = True,
                      signals: Optional[List[Dict]] = None,
                      save_path: Optional[str] = None) -> go.Figure:
        """
        Draw main chart with FVG zones

        Args:
            data: DataFrame OHLC
            fvgs: List of FVGs to plot
            title: Chart title
            show_volume: Show volume subplot
            signals: List of signals to mark (optional)
            save_path: Path to save HTML (optional)

        Returns:
            go.Figure: Plotly figure object
        """

        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(title, 'Volume'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()

        # ===== STEP 1: Plot candlestick =====
        candlestick = go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            showlegend=True
        )

        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)

        # ===== STEP 2: Plot FVG zones =====
        for fvg in fvgs:
            # Skip touched FVGs if not showing
            if fvg.is_touched and not self.show_touched_fvgs:
                continue

            self._add_fvg_zone(fig, data, fvg, row=1 if show_volume else None)

        # ===== STEP 3: Plot signals (if any) =====
        if signals:
            self._add_signals(fig, signals, row=1 if show_volume else None)

        # ===== STEP 4: Plot volume (if enabled) =====
        if show_volume and 'volume' in data.columns:
            colors = ['red' if close < open_ else 'green'
                     for close, open_ in zip(data['close'], data['open'])]

            volume_bars = go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            )
            fig.add_trace(volume_bars, row=2, col=1)

        # ===== STEP 5: Layout configuration =====
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=800 if show_volume else 600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # ===== STEP 6: Save HTML (if specified) =====
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"Chart saved to: {save_path}")

        return fig

    def _add_fvg_zone(self, fig: go.Figure, data: pd.DataFrame, fvg: FVG, row: Optional[int] = None):
        """
        Add FVG zone to chart

        Args:
            fig: Plotly figure
            data: DataFrame OHLC
            fvg: FVG object
            row: Row number (if using subplots)
        """

        # Determine colors
        if fvg.is_touched:
            if fvg.fvg_type == 'BULLISH':
                fill_color = self.colors['bullish_touched']
            else:
                fill_color = self.colors['bearish_touched']
            border_color = self.colors['touched_border']
            line_dash = 'dash'
        else:
            if fvg.fvg_type == 'BULLISH':
                fill_color = self.colors['bullish_active']
                border_color = self.colors['bullish_border']
            else:
                fill_color = self.colors['bearish_active']
                border_color = self.colors['bearish_border']
            line_dash = 'solid'

        # Get time range
        start_time = fvg.created_timestamp
        if fvg.is_touched and fvg.touched_timestamp:
            end_time = fvg.touched_timestamp
        else:
            end_time = data.index[-1]

        # Create rectangle shape
        shape = dict(
            type="rect",
            xref="x",
            yref="y",
            x0=start_time,
            y0=fvg.bottom,
            x1=end_time,
            y1=fvg.top,
            fillcolor=fill_color,
            line=dict(color=border_color, width=1, dash=line_dash),
            layer="below"
        )

        # Add shape to figure
        if row:
            fig.add_shape(shape, row=row, col=1)
        else:
            fig.add_shape(shape)

        # Add label (if enabled)
        if self.show_labels:
            label_text = f"{fvg.fvg_type}<br>Gap: {fvg.gap_size:.5f}<br>Strength: {fvg.strength:.2f}"
            status = "TOUCHED" if fvg.is_touched else "ACTIVE"
            label_text += f"<br>{status}"

            annotation = dict(
                x=start_time,
                y=fvg.middle,
                text=label_text,
                showarrow=False,
                font=dict(size=8, color='white'),
                bgcolor=border_color,
                opacity=0.7,
                xanchor='left'
            )

            if row:
                fig.add_annotation(annotation, row=row, col=1)
            else:
                fig.add_annotation(annotation)

    def _add_signals(self, fig: go.Figure, signals: List[Dict], row: Optional[int] = None):
        """
        Add signal markers to chart

        Args:
            fig: Plotly figure
            signals: List of signal dicts
            row: Row number (if using subplots)
        """

        buy_signals = [s for s in signals if s.get('signal') == 'BUY']
        sell_signals = [s for s in signals if s.get('signal') == 'SELL']

        # Plot BUY signals
        if buy_signals:
            buy_times = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['entry'] for s in buy_signals]

            buy_trace = go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                name='BUY Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='lime',
                    line=dict(color='white', width=1)
                )
            )

            if row:
                fig.add_trace(buy_trace, row=row, col=1)
            else:
                fig.add_trace(buy_trace)

        # Plot SELL signals
        if sell_signals:
            sell_times = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['entry'] for s in sell_signals]

            sell_trace = go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                name='SELL Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(color='white', width=1)
                )
            )

            if row:
                fig.add_trace(sell_trace, row=row, col=1)
            else:
                fig.add_trace(sell_trace)

    def plot_fvg_statistics(self, fvgs: List[FVG], save_path: Optional[str] = None) -> go.Figure:
        """
        Draw FVG statistics charts

        Args:
            fvgs: List of FVGs
            save_path: Path to save HTML (optional)

        Returns:
            go.Figure: Plotly figure
        """

        # Prepare data
        bullish_fvgs = [f for f in fvgs if f.fvg_type == 'BULLISH']
        bearish_fvgs = [f for f in fvgs if f.fvg_type == 'BEARISH']

        bullish_touched = sum(1 for f in bullish_fvgs if f.is_touched)
        bearish_touched = sum(1 for f in bearish_fvgs if f.is_touched)

        bullish_active = len(bullish_fvgs) - bullish_touched
        bearish_active = len(bearish_fvgs) - bearish_touched

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'FVG Count by Type',
                'FVG Status',
                'Gap Size Distribution',
                'FVG Strength Distribution'
            ),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'histogram'}, {'type': 'histogram'}]]
        )

        # 1. FVG Count by Type
        fig.add_trace(
            go.Bar(
                x=['Bullish', 'Bearish'],
                y=[len(bullish_fvgs), len(bearish_fvgs)],
                marker_color=['green', 'red'],
                name='Total FVGs'
            ),
            row=1, col=1
        )

        # 2. FVG Status (Pie chart)
        fig.add_trace(
            go.Pie(
                labels=['Bullish Active', 'Bullish Touched', 'Bearish Active', 'Bearish Touched'],
                values=[bullish_active, bullish_touched, bearish_active, bearish_touched],
                marker_colors=['lightgreen', 'lightgray', 'lightcoral', 'darkgray']
            ),
            row=1, col=2
        )

        # 3. Gap Size Distribution
        gap_sizes = [f.gap_size for f in fvgs]
        fig.add_trace(
            go.Histogram(
                x=gap_sizes,
                nbinsx=30,
                name='Gap Size',
                marker_color='blue'
            ),
            row=2, col=1
        )

        # 4. FVG Strength Distribution
        strengths = [f.strength for f in fvgs]
        fig.add_trace(
            go.Histogram(
                x=strengths,
                nbinsx=30,
                name='Strength',
                marker_color='orange'
            ),
            row=2, col=2
        )

        # Layout
        fig.update_layout(
            title='FVG Statistics Dashboard',
            template='plotly_dark',
            height=800,
            showlegend=False
        )

        # Save if needed
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"Statistics chart saved to: {save_path}")

        return fig

    def create_fvg_report(self, data: pd.DataFrame, fvgs: List[FVG],
                         signals: Optional[List[Dict]] = None,
                         output_dir: str = "logs/charts") -> Dict[str, str]:
        """
        Create full report with multiple charts

        Args:
            data: DataFrame OHLC
            fvgs: List of FVGs
            signals: Signals (optional)
            output_dir: Output directory

        Returns:
            dict: Paths to generated files
        """

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files = {}

        # 1. Main FVG chart
        main_chart_path = os.path.join(output_dir, f'fvg_chart_{timestamp}.html')
        self.plot_fvg_chart(data, fvgs, signals=signals, save_path=main_chart_path)
        files['main_chart'] = main_chart_path

        # 2. Statistics chart
        stats_chart_path = os.path.join(output_dir, f'fvg_statistics_{timestamp}.html')
        self.plot_fvg_statistics(fvgs, save_path=stats_chart_path)
        files['statistics'] = stats_chart_path

        print(f"\nFVG Report created:")
        print(f"  Main Chart: {main_chart_path}")
        print(f"  Statistics: {stats_chart_path}")

        return files


# ===== UTILITY FUNCTIONS =====

def quick_plot_fvgs(data: pd.DataFrame, fvgs: List[FVG],
                   title: str = "Quick FVG View",
                   save_path: Optional[str] = None):
    """
    Quick helper function to plot FVGs

    Args:
        data: DataFrame OHLC
        fvgs: List of FVGs
        title: Chart title
        save_path: Save path (optional)
    """
    visualizer = FVGVisualizer()
    fig = visualizer.plot_fvg_chart(data, fvgs, title=title, save_path=save_path)
    fig.show()


def compare_fvg_periods(data: pd.DataFrame, fvgs_period1: List[FVG],
                       fvgs_period2: List[FVG], period1_name: str = "Period 1",
                       period2_name: str = "Period 2") -> go.Figure:
    """
    Compare FVGs between 2 periods

    Args:
        data: DataFrame OHLC
        fvgs_period1: FVGs from period 1
        fvgs_period2: FVGs from period 2
        period1_name: Name of period 1
        period2_name: Name of period 2

    Returns:
        go.Figure: Comparison chart
    """

    # Create comparison stats
    stats1 = {
        'total': len(fvgs_period1),
        'bullish': sum(1 for f in fvgs_period1 if f.fvg_type == 'BULLISH'),
        'bearish': sum(1 for f in fvgs_period1 if f.fvg_type == 'BEARISH'),
        'touched': sum(1 for f in fvgs_period1 if f.is_touched)
    }

    stats2 = {
        'total': len(fvgs_period2),
        'bullish': sum(1 for f in fvgs_period2 if f.fvg_type == 'BULLISH'),
        'bearish': sum(1 for f in fvgs_period2 if f.fvg_type == 'BEARISH'),
        'touched': sum(1 for f in fvgs_period2 if f.is_touched)
    }

    # Create bar chart
    fig = go.Figure()

    categories = ['Total', 'Bullish', 'Bearish', 'Touched']

    fig.add_trace(go.Bar(
        name=period1_name,
        x=categories,
        y=[stats1['total'], stats1['bullish'], stats1['bearish'], stats1['touched']],
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        name=period2_name,
        x=categories,
        y=[stats2['total'], stats2['bullish'], stats2['bearish'], stats2['touched']],
        marker_color='orange'
    ))

    fig.update_layout(
        title='FVG Comparison',
        xaxis_title='Metric',
        yaxis_title='Count',
        barmode='group',
        template='plotly_dark'
    )

    return fig
