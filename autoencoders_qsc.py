import os
import sys
import io
import base64
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.preprocessing import StandardScaler
import hdbscan
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, LinearColorMapper, ImageURL
from bokeh.palettes import Viridis256
from bokeh.io import output_notebook, export_png
from qsc import Qsc
from sklearn.neural_network import MLPRegressor

params = {'perplexity': 5, 'min_cluster_size': 15, 'ntheta': 60, 'nphi': 110, 'ntheta_fourier': 20, 'radii_to_try_qsc': [0.16, 0.14, 0.12, 0.1, 0.08, 0.05, 0.03], 'zoom_qsc': 1.3, 'nphi_qsc': 151, 'highlight_color': 'red', 'results_path': 'results', 'data_path': 'data'}

def create_3d_image(stel, ntheta, nphi, ntheta_fourier, radii_to_try_qsc, zoom_qsc):
    for r in radii_to_try_qsc:
        try:
            x_2D, y_2D, z_2D, _ = stel.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
            break
        except Exception as e:
            pass
    else:
        raise ValueError("Failed to get boundary")
    
    phi2D, theta2D = np.meshgrid(np.linspace(0, 2 * np.pi, nphi), np.linspace(0, 2 * np.pi, ntheta))
    Bmag = stel.B_mag(r, theta2D, phi2D)
    cmap_plot = LightSource(azdeg=0, altdeg=10).shade(Bmag, cm.viridis, norm=Normalize(vmin=Bmag.min(), vmax=Bmag.max()))
    
    fig = plt.figure(figsize=(3, 3), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_2D, y_2D, z_2D, facecolors=cmap_plot, rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=0.9, shade=False)
    ax.set_box_aspect((np.ptp(x_2D), np.ptp(y_2D), np.ptp(z_2D)), zoom=zoom_qsc)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert('RGBA')
    data = np.array(img)
    data[(data[:, :, :3] == 255).all(axis=2)] = [255, 255, 255, 0]
    img = Image.fromarray(data, mode='RGBA')

    buf = io.BytesIO()
    img.save(buf, format='png')
    buf.seek(0)

    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')

this_path = str(Path(__file__).parent.resolve())
os.chdir(this_path)
nfp = int(sys.argv[1])
filename = os.path.join(this_path, params['data_path'], f'qsc_out.random_scan_nfp{nfp}.csv')
df = pd.read_csv(filename)
df['y'] = df.filter(like='y', axis=1).sum(axis=1)
rel_cols = [col for col in df.columns if col.startswith('x') or col == 'y']
df_rel = df.loc[:, rel_cols]
results_path = os.path.join(this_path, params['results_path']+f'_nfp{nfp}')
os.makedirs(results_path, exist_ok=True)

X = StandardScaler().fit_transform(df_rel.drop(columns=['y']))
y = df_rel['y']

autoencoder = MLPRegressor(hidden_layer_sizes=(50, 2, 50), random_state=42, max_iter=500)
autoencoder.fit(X, X)
X_encoded = autoencoder.transform(X)

params['min_cluster_size'] = max(params['min_cluster_size'], int(len(X_encoded) / 40))
labels = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], min_samples=1).fit_predict(X_encoded)
cluster_means = {label: np.mean(y[labels == label]) for label in np.unique(labels)}
min_y_mean_cluster = min(cluster_means, key=cluster_means.get)

source = ColumnDataSource(data=dict(x=X_encoded[:, 0], y=X_encoded[:, 1], cluster=[str(label) for label in labels], y_mean=[cluster_means[label] for label in labels], is_lowest_mean=[label == min_y_mean_cluster for label in labels]))
color_mapper = LinearColorMapper(palette=Viridis256, low=min(labels), high=max(labels))
output_notebook()
p = figure(width=800, height=600, title=rf"Autoencoder Clustering of nfp={nfp} using y mean", tools="pan,wheel_zoom,box_zoom,reset", output_backend="canvas")
p.add_tools(HoverTool(tooltips=[("Cluster", "@cluster"), ("Mean of y", "@y_mean")]))
p.scatter('x', 'y', source=source, legend_field='cluster', color={'field': 'cluster', 'transform': color_mapper}, size=3, alpha=0.5)
p.scatter('x', 'y', source=source, color=params['highlight_color'], size='is_lowest_mean', line_color=params['highlight_color'])
p.xaxis.axis_label = 'Autoencoder 1'
p.yaxis.axis_label = 'Autoencoder 2'
p.xaxis.axis_label_text_font_size = "18pt"
p.yaxis.axis_label_text_font_size = "18pt"
p.xaxis.major_label_text_font_size = "14pt"
p.yaxis.major_label_text_font_size = "14pt"
p.title.text_font_size = "18pt"

for count, label in enumerate(np.unique(labels)):
    selected_indices = np.where((labels == label) & (y == min(y[labels == label])))[0]
    x_selected = df.loc[selected_indices, [col for col in df.columns if col.startswith('x')]]
    x_values = x_selected.filter(like='x', axis=1).iloc[:, :10].values.flatten()
    rc, zs = np.insert(x_values[::2], 0, 1), np.insert(x_values[1::2], 0, 0)
    etabar, B2c = x_selected.loc[:, 'x11'].values[0], x_selected.loc[:, 'x12'].values[0]
    stel = Qsc(rc=rc.tolist(), zs=zs.tolist(), nfp=nfp, etabar=etabar, order='r3', B2c=B2c, nphi=params['nphi_qsc'])

    mean_x, mean_y = np.mean(X_encoded[labels == label, 0]) - 7, np.mean(X_encoded[labels == label, 1]) + 7

    try:
        image_uri = create_3d_image(stel, params['ntheta'], params['nphi'], params['ntheta_fourier'], params['radii_to_try_qsc'], params['zoom_qsc'])
        image_url_source = ColumnDataSource(data=dict(url=[image_uri]))
        image_glyph = ImageURL(url="url", x=mean_x, y=mean_y, w=15, h=15)
        p.add_glyph(image_url_source, image_glyph)
    except Exception as e:
        print(e)
        pass

p.toolbar_location = None
export_png(p, filename=os.path.join(results_path, f"autoencoder_stellarators_nfp{nfp}.png"))
