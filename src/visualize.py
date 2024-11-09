import plotly.graph_objects as go
import numpy as np

def plot_segment(times: np.ndarray, data: np.ndarray, label: int, pred: int, segment_type: str):
    fig = go.Figure()
    # Добавление каналов
    for i in range(data.shape[0]):
        fig.add_trace(go.Scatter(x=times, y=data[i], name=f'Канал {i+1}'))
    
    fig.update_layout(
        title_text=f"Визуализация сегмента: {segment_type}. Предсказание: {pred}. Правильный ответ: {label}",
        showlegend=True,
        height=500,
        width=1000,
        xaxis_title='Время (с)', 
        yaxis_title='Амплитуда'
    )
    
    fig.show()