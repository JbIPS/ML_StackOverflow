# StackOverflow Tag prediction

## Organisation
Ce dépôt contient deux notebook Jupyter ainsi qu'une application de prédiction.

### P5_notebook_exploration.ipynb
Ce notebook contient une analyse du dataset sur la fréquence des mots et des tags, ainsi qu'un nettoyage des balises HTML.

Il exporte un dataset utilisé par le deuxième notebook.

### P5_notebook_test.ipynb
Ce notebook test différents modèles de classification multilabels, sélectionne le meilleur, effectue une recherche d'hyperparamètres et sérialise le meilleur modèle afin qu'il soit utilisé dans l'application.

### app/
L'application de prédiction comporte un frontend Streamlit permettant d'afficher une interface sommaire pour y entrer un texte à analyser, ainsi qu'un backend FastAPI pour exposer le modèle grâce à une API REST.

## Installation
L'application peut se lancer grâce à la commande `docker-compose`:
```
docker-compose up
```
lancée à la racine (là où se trouve le fichier `docker-compose.yml`).

## Utilisation
### Frontend
Une fois le swarm docker lancé, l'UI de l'application est utilisable à l'adresse http://localhost:8501.

### API
L'API est utilisable indépendemment à l'adresse http://localhost:8000. La spécification OpenAPI est disponible [ici](http://localhost:8000/docs).
