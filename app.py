from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

df = pd.read_csv('mov.csv')  
df['release_year'] = pd.to_datetime(df['release_date']).dt.year

def create_bins(sub_df):
    # vote bins (0-1, 1-2, ... , 9-10 )
    vote_bins = np.arange(0, 11, 1)
    vote_labels = [f'{i}-{i+1}' for i in range(0, 10)]
    sub_df['vote_bin'] = pd.cut(sub_df['vote_average'], bins=vote_bins, labels=vote_labels, include_lowest=True, right=False)

    # year bins
    min_year = sub_df['release_year'].min()
    max_year = sub_df['release_year'].max()

    if min_year == max_year:
        year_bins = [min_year, min_year + 1]
    else:
        raw_bins = np.linspace(min_year, max_year + 1, num=11, dtype=int)
        year_bins = np.unique(raw_bins)  # no duplicates
        if len(year_bins) < 2:
            year_bins = [min_year, max_year + 1]

    # generate labels
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

    if len(filtered_df) <= 100:
        movies = filtered_df[['title', 'vote_average', 'release_year']].sort_values(by='vote_average', ascending=False)
        return render_template('grid_detail.html', movies=movies, vote_bin=v_filter, year_bin=y_filter)

    binned_df, vote_bins, year_bins = create_bins(filtered_df)
    pivot = binned_df.pivot_table(index='vote_bin', columns='year_bin', aggfunc='size', fill_value=0)
    pivot = pivot.reindex(index=vote_bins, columns=year_bins, fill_value=0)
    return render_template('grid.html', vote_bins=vote_bins, year_bins=year_bins, grid=pivot.values.tolist(), vote_filter=v_filter, year_filter=y_filter)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
