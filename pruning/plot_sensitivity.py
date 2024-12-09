import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go


results = dict(np.load('sensitivity.npz'))
ratios = results.pop('ratios')

num_modules = len(results)
cols = 3
rows = (num_modules + cols - 1) // cols

titles = list(results.keys())
for i, title in enumerate(titles):
    if ',' in title:
        titles[i] = 'grouped ' + title[:15]

fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                       vertical_spacing=0.015, horizontal_spacing=0.04)

for i, (module_name, metrics) in enumerate(results.items()):
    row = i // cols + 1
    col = i % cols + 1

    fig.add_trace(
        go.Scatter(x=ratios, y=metrics, mode='lines+markers', name=module_name, line=dict(color='black')),
        row=row, col=col)

    if col == 1:
        fig.update_yaxes(title_text='Mel Spec. Error', row=row, col=col)

fig.update_layout(
    height=200 * rows,
    width=1000,
    title='Pruning Sensitivity Analysis',
    showlegend=False
)

fig.write_image('sensitivity.pdf', format='pdf')
