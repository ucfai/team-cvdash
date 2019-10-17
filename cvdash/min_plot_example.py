import plotly.offline as py
import plotly.graph_objects as go

x = [20, 14, 23]
y = ['giraffes', 'orangutans', 'monkeys']

fig = go.Figure(go.Bar(
            x=x,
            y=y,
            orientation='h'))

annotations = []

for xi, yi in zip(x, y):
    annotations.append(dict(xref='x1', yref='y1',
                            y=yi, x=xi + 1,
                            text=str(xi),
                            font=dict(family='Arial', size=20,
                                      color='blue'),
                            showarrow=False))

fig.update_layout(annotations=annotations)
fig.show()