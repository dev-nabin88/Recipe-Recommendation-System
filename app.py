from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='error.log', level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset (already filtered for Indian cuisine)
try:
    data = pd.read_csv("modified.csv")
    indian_data = data[['recipe_name', 'ingredients_list', 'image_url', 'diet', 'state']].dropna()
    indian_data['diet'] = indian_data['diet'].str.strip().str.lower()
    indian_data['state'] = indian_data['state'].str.strip().str.lower()
except Exception as e:
    logging.warning(f"Error loading dataset: {e}")
    indian_data = pd.DataFrame(columns=['recipe_name', 'ingredients_list', 'image_url', 'diet', 'state'])

# Preprocess Ingredients
try:
    vectorizer = TfidfVectorizer()
    if not indian_data.empty:
        X_ingredients = vectorizer.fit_transform(indian_data['ingredients_list'])
        knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        knn.fit(X_ingredients)
    else:
        X_ingredients = None
        knn = None
except Exception as e:
    logging.warning(f"Error initializing vectorizer or KNN: {e}")
    X_ingredients = None
    knn = None

# Validate user input for ingredients
def validate_ingredients(input_ingredients):
    pattern = re.compile(r'^[a-zA-Z,\s]+$')
    if not pattern.match(input_ingredients):
        return False  # Invalid input
    return True

# Clean user input for ingredients
def clean_ingredients(input_ingredients):
    cleaned = re.sub(r'[^a-zA-Z,\s]', '', input_ingredients).strip()
    return cleaned

# Check ingredient compatibility with diet
def is_compatible_with_diet(ingredients, diet):
    non_veg_keywords = ['chicken', 'mutton', 'fish', 'egg', 'prawn', 'beef', 'pork']
    if diet == 'veg':
        for keyword in non_veg_keywords:
            if keyword in ingredients.lower():
                return False
    return True

# Recommendation function
def recommend_recipes(input_ingredients, selected_diet, selected_state):
    try:
        # Filter data based on user input
        filtered_data = indian_data
        if selected_diet:
            filtered_data = filtered_data[filtered_data['diet'] == selected_diet.strip().lower()]
        if selected_state:
            filtered_data = filtered_data[filtered_data['state'] == selected_state.strip().lower()]

        # Handle empty filtered data
        if filtered_data.empty:
            return None

        # Transform input ingredients and predict
        filtered_ingredients = vectorizer.transform(filtered_data['ingredients_list'])
        input_ingredients_transformed = vectorizer.transform([input_ingredients])
        knn.fit(filtered_ingredients)  # Fit KNN on the filtered data
        distances, indices = knn.kneighbors(input_ingredients_transformed)
        recommendations = filtered_data.iloc[indices[0]]
        return recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(5)
    except Exception as e:
        logging.warning(f"Error in recommendation system: {e}")
        return None

# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        diets = indian_data['diet'].dropna().unique() if not indian_data.empty else []
        states = indian_data['state'].dropna().unique() if not indian_data.empty else []

        if request.method == 'POST':
            ingredients = request.form.get('ingredients', '').strip()
            selected_diet = request.form.get('diet')
            selected_state = request.form.get('state')

            # Validate and clean input
            if not validate_ingredients(ingredients):
                return render_template('index.html', 
                                       error="Invalid input. Please enter valid ingredients (e.g., 'tomato, onion').", 
                                       diets=diets, 
                                       states=states, 
                                       truncate=truncate)

            ingredients = clean_ingredients(ingredients)
            if not ingredients:
                return render_template('index.html', 
                                       error="Ingredients cannot be empty or invalid after cleaning.", 
                                       diets=diets, 
                                       states=states, 
                                       truncate=truncate)

            # Check compatibility of ingredients with selected diet
            if selected_diet and not is_compatible_with_diet(ingredients, selected_diet.strip().lower()):
                return render_template('index.html', 
                                       error="No recipes found for the given input.", 
                                       diets=diets, 
                                       states=states, 
                                       truncate=truncate)

            recommendations = recommend_recipes(ingredients, selected_diet, selected_state)

            # Show message if no recommendations found
            if recommendations is None or recommendations.empty:
                return render_template('index.html', 
                                       error="No recipes found for the given input.", 
                                       diets=diets, 
                                       states=states, 
                                       truncate=truncate)

            return render_template('index.html', 
                                   recommendations=recommendations.to_dict(orient='records'), 
                                   diets=diets, 
                                   states=states, 
                                   truncate=truncate)

        return render_template('index.html', recommendations=[], diets=diets, states=states)
    except Exception as e:
        logging.warning(f"Error in index route: {e}")
        return render_template('index.html', error="An unexpected error occurred.", diets=[], states=[], truncate=truncate)

# Additional routes for specific recipe categories
@app.route('/<category>', methods=['GET', 'POST'])
def category_page(category):
    try:
        template_name = f"{category}.html"
        return render_template(template_name)
    except Exception as e:
        logging.warning(f"Error loading category page '{category}': {e}")
        return render_template('404.html', error=f"Category '{category}' not found.")
@app.route('/index', methods=['GET','POST'])
def home():
    return render_template('index.html')


@app.route('/main-course', methods=['GET','POST'])
def mainCourse():
    return render_template('main-course.html')


@app.route('/appetizers', methods=['GET','POST'])
def appetizers():
    return render_template('appetizers.html')


@app.route('/dessarts', methods=['GET','POST'])
def dessarts():
    return render_template('dessarts.html')


@app.route('/drinks', methods=['GET','POST'])
def drinks():
    return render_template('drinks.html')


@app.route('/momo', methods=['GET','POST'])
def momo():
    return render_template('momo.html')


@app.route('/samosa', methods=['GET','POST'])
def samosa():
    return render_template('samosa.html')


@app.route('/pakora', methods=['GET','POST'])
def pakora():
    return render_template('pakora.html')
@app.route('/bhelpuri', methods=['GET','POST'])
def bhelpuri():
    return render_template('bhelpuri.html')


@app.route('/paneer', methods=['GET','POST'])
def paneer():
    return render_template('paneer.html')


@app.route('/pulao', methods=['GET','POST'])
def pulao():
    return render_template('pulao.html')


@app.route('/mutton', methods=['GET','POST'])
def mutton():
    return render_template('mutton.html')
@app.route('/contribute', methods=['GET','POST'])
def contribute():
    return render_template('contribute.html')
@app.route('/gulabjamun', methods=['GET','POST'])
def gulabjamun():
    return render_template('gulabjamun.html')
@app.route('/rasgulla', methods=['GET','POST'])
def rasgulla():
    return render_template('rasgulla.html')
@app.route('/kalakand', methods=['GET','POST'])
def kalakand():
    return render_template('kalakand.html')
@app.route('/sandesh', methods=['GET','POST'])
def sandesh():
    return render_template('sandesh.html')
@app.route('/ampanna', methods=['GET','POST'])
def ampanna():
    return render_template('ampanna.html')
@app.route('/malai', methods=['GET','POST'])
def malai():
    return render_template('malai.html')
@app.route('/jaljeera', methods=['GET','POST'])
def jaljeera():
    return render_template('jaljeera.html')
@app.route('/lassi', methods=['GET','POST'])
def lassi():
    return render_template('lassi.html')

if __name__ == '__main__':
    app.run(debug=True)
