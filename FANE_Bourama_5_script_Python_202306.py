import pandas as pd  # Importation de la bibliothèque pandas pour manipuler les données
import json  # Importation de la bibliothèque json pour traiter les réponses JSON
import requests  # Importation de la bibliothèque requests pour effectuer des requêtes HTTP

# Définition de l'URL de l'API Edamam Food Database pour la recherche d'aliments
url = "https://edamam-food-and-grocery-database.p.rapidapi.com/api/food-database/v2/parser"

# Paramètres de la requête à l'API Edamam Food Database
querystring = {
    "app_key": "56c9423f7a3c76f991e97c781c7ed958",
    "app_id": "bf70547b",
    "nutrition-type": "cooking",
    "category[0]": "generic-foods",
    "health[0]": "alcohol-free",
    "ingr": "champagne"
}

# En-têtes de la requête HTTP

headers = {
    "X-RapidAPI-Key": "2912abea9cmsh9bf3fe4dc41b3b4p10a6c7jsnc3a9e4894b9f",
    "X-RapidAPI-Host": "edamam-food-and-grocery-database.p.rapidapi.com"
}

# Envoi de la requête GET à l'API Edamam Food Database
response = requests.get(url, headers=headers, params=querystring)

# Extraction de la réponse JSON de la requête
data_API = response.json()

# Extraction de la liste des suggestions d'aliments à partir des résultats JSON
hints = data_API.get('hints', [])

# Liste pour stocker les informations des suggestions d'aliments foodId, label, category,
# foodContentsLabel, image.

hint_data = []
for hint in hints:
    food = hint['food']
    food_id = food.get('foodId')
    label = food.get('label')
    category = food.get('category')
    food_contents_label = food.get('foodContentsLabel')
    image = food.get('image')

    hint_data.append({
        'foodId': food_id,
        'label': label,
        'category': category,
        'foodContentsLabel': food_contents_label,
        'image': image
    })

# Vérification si des données ont été extraites
if hint_data:
    # Création d'un DataFrame pandas à partir des 10 premières suggestions d'aliments
    df_API = pd.DataFrame(hint_data[:10])

# Sauvegarde du DataFrame dans un fichier CSV appelé "API_Champagne.csv" sans inclure l'index
df_API.to_csv('APi_Champagne.csv', index=False)

print(df_API)
