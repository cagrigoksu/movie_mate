from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sqlite3
import requests
from bs4 import BeautifulSoup
import json

app = Flask(__name__)

DB_FILE = 'movies_final.db'

def get_connection():
    return sqlite3.connect(DB_FILE)

def load_movies():
    conn = get_connection()
    query = '''
        SELECT m.id, m.title, m.vote_average, m.release_date,
               GROUP_CONCAT(g.name, ', ') AS genres
        FROM movies m
        LEFT JOIN movie_genres mg ON m.id = mg.movie_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE m.release_date IS NOT NULL AND m.vote_average IS NOT NULL
        GROUP BY m.id
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df = df.dropna(subset=['release_year'])
    df['release_year'] = df['release_year'].astype(int)

    return df

def get_all_genres():
    conn = get_connection()
    query = 'SELECT DISTINCT name FROM genres ORDER BY name ASC'
    genres = pd.read_sql_query(query, conn)['name'].tolist()
    conn.close()
    return genres

def create_bins(sub_df):
    min_vote = sub_df['vote_average'].min()
    max_vote = sub_df['vote_average'].max()
    min_year = sub_df['release_year'].min()
    max_year = sub_df['release_year'].max()

    vote_bins = np.arange(0, 11, 1)
    vote_labels = [f'{i}-{i+1}' for i in range(0, 10)]
    sub_df['vote_bin'] = pd.cut(sub_df['vote_average'], bins=vote_bins, labels=vote_labels, include_lowest=True, right=False)

    raw_bins = np.linspace(min_year, max_year + 1, num=11, dtype=int)
    year_bins = np.unique(raw_bins)
    year_labels = [f'{year_bins[i]}-{year_bins[i+1]-1}' for i in range(len(year_bins) - 1)]
    sub_df['year_bin'] = pd.cut(sub_df['release_year'], bins=year_bins, labels=year_labels, include_lowest=True, right=False)

    return sub_df, vote_labels[::-1], year_labels

df = load_movies()

@app.route('/')
@app.route('/grid')
def grid():
    v_filter = request.args.get('v')
    y_filter = request.args.get('y')
    min_rating = request.args.get('min_rating', type=float)
    max_rating = request.args.get('max_rating', type=float)
    min_year = request.args.get('min_year', type=int)
    max_year = request.args.get('max_year', type=int)
    genres_selected = request.args.getlist('genres')

    filtered_df = df.copy()

    # Global year bounds
    global_min_year = df['release_year'].min()
    global_max_year = df['release_year'].max()

    if min_rating is not None:
        filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]
    if max_rating is not None:
        filtered_df = filtered_df[filtered_df['vote_average'] <= max_rating]
    if min_year is not None:
        filtered_df = filtered_df[filtered_df['release_year'] >= min_year]
    if max_year is not None:
        filtered_df = filtered_df[filtered_df['release_year'] <= max_year]
    if genres_selected:
        filtered_df = filtered_df[
            filtered_df['genres'].apply(
                lambda g: bool(g) and isinstance(g, str) and any(genre in g.split(', ') for genre in genres_selected)
            )
        ]

    if v_filter and y_filter:
        v_low, v_high = map(float, v_filter.split('-'))
        y_low, y_high = map(int, y_filter.split('-'))

        filtered_df = filtered_df[
            (filtered_df['vote_average'] >= v_low) & (filtered_df['vote_average'] < v_high) &
            (filtered_df['release_year'] >= y_low) & (filtered_df['release_year'] <= y_high)
        ]

        vote_range = v_high - v_low
        year_range = y_high - y_low

        is_fine_zoom = (vote_range <= 1 and '.' in v_filter) and (year_range == 0)

        if is_fine_zoom or len(filtered_df) <= 100:
            movies = filtered_df[['id', 'title', 'vote_average', 'release_year']].sort_values(by='vote_average', ascending=False)
            return render_template('grid_detail.html', movies=movies, vote_bin=v_filter, year_bin=y_filter)

    binned_df, vote_bins, year_bins = create_bins(filtered_df)
    pivot = binned_df.pivot_table(index='vote_bin', columns='year_bin', aggfunc='size', fill_value=0)
    pivot = pivot.reindex(index=vote_bins, columns=year_bins, fill_value=0)

    return render_template('grid.html',
                           vote_bins=vote_bins,
                           year_bins=year_bins,
                           grid=pivot.values.tolist(),
                           vote_filter=v_filter,
                           year_filter=y_filter,
                           all_genres=get_all_genres(),
                           global_min_year=global_min_year,
                           global_max_year=global_max_year)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    conn = get_connection()

    movie_query = '''
        SELECT id, title, overview, release_date, vote_average, runtime, poster_path, imdb_id
        FROM movies
        WHERE id = ?
    '''
    movie = pd.read_sql_query(movie_query, conn, params=(movie_id,)).iloc[0]

    conn.close()

    imdb_rating = None
    imdb_poster = None
    
    if pd.notnull(movie['poster_path']) or movie['poster_path']=="":

        if pd.notnull(movie['imdb_id']):
            imdb_url = f"https://www.imdb.com/title/{movie['imdb_id']}/"
            headers = {
                'User-Agent': 'Mozilla/5.0'
            }

            try:
                response = requests.get(imdb_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    script_tag = soup.find('script', type='application/ld+json')
                    if script_tag:
                    
                        json_data = json.loads(script_tag.string)

                        if 'aggregateRating' in json_data:
                            imdb_rating = json_data['aggregateRating'].get('ratingValue', None)
                        
                        if 'image' in json_data:
                            imdb_poster = json_data['image']

            except Exception as e:
                print(f"IMDb scraping failed: {e}")

    else:
        imdb_rating = movie['vote average']
        imdb_poster = movie['poster_path']
        
    movie_data = {
        'id': movie['id'],
        'title': movie['title'],
        'overview': movie['overview'],
        'release_year': pd.to_datetime(movie['release_date']).year if pd.notnull(movie['release_date']) else '',
        'vote_average': movie['vote_average'],
        'runtime': movie['runtime'],
        'poster_path': movie['poster_path'],
        'imdb_id': movie['imdb_id'],
        'imdb_rating': imdb_rating,
        'imdb_poster': imdb_poster
    }

    return render_template('movie_detail.html', movie=movie_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
