from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("mov.csv")
df['release_date'] = pd.to_datetime(df['release_date'])
df['year'] = df['release_date'].dt.year

# Bin settings
grid_size = 10
vote_edges = np.linspace(0, 10, grid_size + 1)
year_edges = np.linspace(df['year'].min(), df['year'].max() + 1, grid_size + 1, dtype=int)

# Bin assignments
df['x_bin'] = pd.cut(df['year'], bins=year_edges, labels=False, include_lowest=True)
df['y_bin'] = pd.cut(df['vote_average'], bins=vote_edges, labels=False, include_lowest=True)

# Color setup
from matplotlib import cm
from matplotlib.colors import to_hex

cmap_x = cm.Blues
cmap_y = cm.Reds

def get_tile_color(x, y):
    cx = cmap_x(x / (grid_size - 1))  # year bin
    cy = cmap_y(1 - (y / (grid_size - 1)))  # vote bin reversed
    blended = [(cx[i] + cy[i]) / 2 for i in range(3)]
    return to_hex(blended)

@app.route("/")
def index():
    grid = [[{"count": 0, "color": "#ffffff", "x": x, "y": y, "text_color": "#000"} for x in range(grid_size)] for y in range(grid_size)]

    for _, row in df.iterrows():
        x, y = row['x_bin'], row['y_bin']
        if pd.notna(x) and pd.notna(y):
            x, y = int(x), int(y)
            y = grid_size - 1 - y
            grid[y][x]["count"] += 1

    for y in range(grid_size):
        for x in range(grid_size):
            count = grid[y][x]["count"]
            grid[y][x]["color"] = get_tile_color(x, y)
            grid[y][x]["text_color"] = "#000" if count < 300 else "#fff"

    year_labels = [f"{year_edges[i]}–{year_edges[i+1]-1}" for i in range(grid_size)]
    vote_labels = [f"{vote_edges[grid_size - i -1]:.1f}–{vote_edges[grid_size - i]:.1f}" for i in range(grid_size)]
    grid_with_votes = list(zip(grid, vote_labels))

    return render_template("grid.html",
                           grid=grid,
                           grid_with_votes=grid_with_votes,
                           year_labels=year_labels,
                           vote_labels=vote_labels,
                           vote_edges=vote_edges,
                           year_edges=year_edges,
                           grid_size=grid_size,
                           zoom=False)

@app.route("/zoom")
def zoom():
    vmin = float(request.args.get("vmin"))
    vmax = float(request.args.get("vmax"))
    ymin = int(request.args.get("ymin"))
    ymax = int(request.args.get("ymax"))

    filtered_df = df[(df["vote_average"] >= vmin) & (df["vote_average"] <= vmax) &
                     (df["year"] >= ymin) & (df["year"] < ymax)]

    if len(filtered_df) <= 100:
        movies = filtered_df.sort_values(by="vote_average", ascending=False)[["title", "vote_average", "year"]]
        return render_template("grid_detail.html", movies=movies, vmin=vmin, vmax=vmax, ymin=ymin, ymax=ymax)

    # Recalculate bins for zoomed view
    vote_edges_zoom = np.linspace(vmin, vmax, grid_size + 1)
    year_edges_zoom = np.linspace(ymin, ymax, grid_size + 1, dtype=int)

    filtered_df['x_bin'] = pd.cut(filtered_df['year'], bins=year_edges_zoom, labels=False, include_lowest=True)
    filtered_df['y_bin'] = pd.cut(filtered_df['vote_average'], bins=vote_edges_zoom, labels=False, include_lowest=True)

    grid = [[{"count": 0, "color": "#ffffff", "x": x, "y": y, "text_color": "#000"} for x in range(grid_size)] for y in range(grid_size)]

    for _, row in filtered_df.iterrows():
        x, y = row['x_bin'], row['y_bin']
        if pd.notna(x) and pd.notna(y):
            x, y = int(x), int(y)
            y = grid_size - 1 - y
            grid[y][x]["count"] += 1

    for y in range(grid_size):
        for x in range(grid_size):
            count = grid[y][x]["count"]
            grid[y][x]["color"] = get_tile_color(x, y)
            grid[y][x]["text_color"] = "#000" if count < 300 else "#fff"

    year_labels = [f"{year_edges_zoom[i]}–{year_edges_zoom[i+1]-1}" for i in range(grid_size)]
    vote_labels = [f"{vote_edges_zoom[grid_size - i -1]:.1f}–{vote_edges_zoom[grid_size - i]:.1f}" for i in range(grid_size)]
    grid_with_votes = list(zip(grid, vote_labels))

    return render_template("grid.html",
                           grid=grid,
                           grid_with_votes=grid_with_votes,
                           year_labels=year_labels,
                           vote_labels=vote_labels,
                           vote_edges=vote_edges_zoom,
                           year_edges=year_edges_zoom,
                           grid_size=grid_size,
                           zoom=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
