from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sqlite3

app = Flask(__name__)

# Connect to the SQLite database
def get_connection():
    return sqlite3.connect('movies_final.db')  # replace with your SQLite DB filename

# Load initial movie data
def load_movies():
    conn = get_connection()
    query = '''
        SELECT id, title, vote_average, release_date
        FROM movies
        WHERE release_date IS NOT NULL AND vote_average IS NOT NULL
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df = df.dropna(subset=['release_year'])  # Drop rows where release_year could not be parsed
    df['release_year'] = df['release_year'].astype(int)

    return df

df = load_movies()

def create_bins(sub_df):
    min_vote = sub_df['vote_average'].min()
    max_vote = sub_df['vote_average'].max()
    min_year = sub_df['release_year'].min()
    max_year = sub_df['release_year'].max()

    if max_vote - min_vote <= 1:
        vote_bins = np.round(np.arange(np.floor(min_vote*10)/10, np.ceil(max_vote*10)/10 + 0.1, 0.1), 2)
        vote_labels = [f'{vote_bins[i]:.1f}-{vote_bins[i+1]:.1f}' for i in range(len(vote_bins) - 1)]
    else:
        vote_bins = np.arange(0, 11, 1)
        vote_labels = [f'{i}-{i+1}' for i in range(0, 10)]

    sub_df['vote_bin'] = pd.cut(sub_df['vote_average'], bins=vote_bins, labels=vote_labels, include_lowest=True, right=False)

    # year binning
    if min_year == max_year:
        year_bins = [min_year, min_year + 1]
    else:
        raw_bins = np.linspace(min_year, max_year + 1, num=11, dtype=int)
        year_bins = np.unique(raw_bins)
        if len(year_bins) < 2:
            year_bins = [min_year, max_year + 1]

    year_labels = [f'{year_bins[i]}-{year_bins[i+1]-1}' for i in range(len(year_bins) - 1)]
    sub_df['year_bin'] = pd.cut(sub_df['release_year'], bins=year_bins, labels=year_labels, include_lowest=True, right=False)

    return sub_df, vote_labels[::-1], year_labels

@app.route('/')
@app.route('/grid')
def grid():
    v_filter = request.args.get('v')  # vote_bin
    y_filter = request.args.get('y')  # year_bin

    filtered_df = df.copy()

    if v_filter and y_filter:
        v_low, v_high = map(float, v_filter.split('-'))
        y_low, y_high = map(int, y_filter.split('-'))
        filtered_df = filtered_df[
            (filtered_df['vote_average'] >= v_low) & (filtered_df['vote_average'] < v_high) &
            (filtered_df['release_year'] >= y_low) & (filtered_df['release_year'] <= y_high)
        ]

        # finest zoom (0.1 bins + single year), show list even if > 100
        vote_range = v_high - v_low
        is_fine_vote_zoom = vote_range <= 1 and '.' in v_filter
        is_single_year = y_low == y_high
        if is_fine_vote_zoom and is_single_year:
            movies = filtered_df[['title', 'vote_average', 'release_year']].sort_values(by='vote_average', ascending=False)
            return render_template('grid_detail.html', movies=movies, vote_bin=v_filter, year_bin=y_filter)

    # movie count <= 100, show movie list
    if len(filtered_df) <= 100:
        movies = filtered_df[['title', 'vote_average', 'release_year']].sort_values(by='vote_average', ascending=False)
        return render_template('grid_detail.html', movies=movies, vote_bin=v_filter, year_bin=y_filter)

    # otherwise show grid
    binned_df, vote_bins, year_bins = create_bins(filtered_df)
    pivot = binned_df.pivot_table(index='vote_bin', columns='year_bin', aggfunc='size', fill_value=0)
    pivot = pivot.reindex(index=vote_bins, columns=year_bins, fill_value=0)

    return render_template('grid.html',
                           vote_bins=vote_bins,
                           year_bins=year_bins,
                           grid=pivot.values.tolist(),
                           vote_filter=v_filter,
                           year_filter=y_filter)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
