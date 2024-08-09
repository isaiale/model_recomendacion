from flask import Flask, render_template, redirect, url_for, session
import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules
import requests

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Consumir la API para obtener los productos
def get_products_from_api():
    url = "https://back-end-enfermera.vercel.app/api/productos/productos-df"
    response = requests.get(url)
    
    if response.status_code == 200:
        products = response.json()
        return pd.DataFrame(products)
    else:
        raise Exception(f"Error al obtener los datos: {response.status_code}")

# Obtener los datos de la API
df = get_products_from_api()

# Codificar las transacciones en un formato binario
def encode_transactions(df):
    transactions = df['nombre'].apply(lambda x: pd.Series(1, index=[x]))
    return transactions.fillna(0).astype(bool)

df_encoded = encode_transactions(df)

# Aplicar el algoritmo Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/catalog')
def catalog():
    products = df.to_dict(orient='records')
    return render_template('catalog.html', products=products)

@app.route('/buy/<product>')
def buy(product):
    if 'cart' not in session:
        session['cart'] = []
    session['cart'].append(product)

    # Obtener recomendaciones basadas en las compras
    recommendations = recommend_products(session['cart'])
    
    selected_product = df[df['nombre'] == product].iloc[0].to_dict()
    return render_template('recommendations.html', product=selected_product, recommendations=recommendations)

def recommend_products(cart):
    recommended = []

    for item in cart:
        # Filtrar el DataFrame para encontrar el producto actual
        matching_products = df[df['nombre'] == item]
        
        # Verificar si el producto existe
        if matching_products.empty:
            print(f"Producto no encontrado: {item}")
            continue
        
        current_product = matching_products.iloc[0]
        
        # Recomendaciones basadas en descuento
        if current_product['descuento'] > 0:
            discounted_products = df[(df['descuento'] > 0) & (df['nombre'] != item)]
            recommended.extend(discounted_products.to_dict('records'))
        
        # Recomendaciones basadas en la misma categoría
        same_category_products = df[(df['categoria'] == current_product['categoria']) & (df['nombre'] != item)]
        recommended.extend(same_category_products.to_dict('records'))

        # Recomendaciones basadas en precios similares (dentro de un rango del 10%)
        price_range = 0.1 * current_product['precio']
        similar_price_products = df[(df['precio'].between(current_product['precio'] - price_range, current_product['precio'] + price_range)) & (df['nombre'] != item)]
        recommended.extend(similar_price_products.to_dict('records'))

    # Eliminar duplicados basados en el nombre del producto
    seen = set()
    unique_recommendations = []
    for rec in recommended:
        if rec['nombre'] not in seen:
            unique_recommendations.append(rec)
            seen.add(rec['nombre'])

    return unique_recommendations


if __name__ == '__main__':
    app.run(debug=True)
