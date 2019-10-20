import plotly.offline as py
import plotly.graph_objects as go

x = [20, 14, 23]
y = ['giraffes', 'orangutans', 'monkeys']


fig = go.Figure(go.Bar(
            x=x,
            y=y,
            orientation='h',
            text = x,
            textfont = dict(size=24),
            textposition='auto'))

fig.show()