import plotly.graph_objects as go
import numpy as np

def plot_segment(times: np.ndarray, data: np.ndarray, rolling_var: np.ndarray, labels: dict, segment_type: str):
    fig = go.Figure()
    # Добавление каналов
    for i in range(data.shape[0]):
        fig.add_trace(go.Scatter(x=times, y=data[i], name=f'Канал {i+1}'))
    
    # Добавление графика дисперсии
    fig.add_trace(go.Scatter(x=times[:len(rolling_var)], y=rolling_var, name='Дисперсия', line=dict(dash='dot')))
    
    # Добавление вертикальных линий для лейблов
    colors = {'swd': 'red', 'is': 'green', 'ds': 'orange'}
    for key, value in labels.items():
        if value:
            fig.add_vline(x=times[len(times)//2], line_dash="dash", line_color=colors[key], annotation_text=key)
    
    fig.update_layout(
        title_text=f"Визуализация сегмента: {segment_type}",
        showlegend=True,
        height=500,
        width=1000,
        xaxis_title='Время (с)', 
        yaxis_title='Амплитуда'
    )
    
    fig.show()