# Recommender-Systems-Project-2024

## Data Collection and Preprocessing
J'ai commencé par télécharger le dataset MovieLens sur le site [MovieLens](https://grouplens.org/datasets/movielens/).
J'ai utilisé le fichier `ml-1M.zip`. Le dataset contient : la liste des films et la liste des notes données par les
utilisateurs. J'ai combiné les deux datasets pour avoir pour chaque lignes le rating d'un user avec le film, l'année et
son genre.

Ensuite j'ai téléchargé le dataset IMDb sur le site [IMDb Datasets](https://developer.imdb.com/non-commercial-datasets/). J'ai usilisé les
dataset `title.basics.tsv.gz`, `title.ratings.tsv.gz` et `title.crew.tsv.gz`. J'ai fusionné les trois datasets pour avoir
pour chaque ligne le film, l'année, sa note moyenne, son genre, le directeur et les scénaristes. J'ai essayé d'ajouter
le cast du films mais le dataset était trop grand donc le temps de téléchargement était trop long.

J'ai fusionné les deux datasets pour avoir un dataset avec pour chaque ligne le user, le films, l'année, le genre, la note
de l'utilisateur, la moyenne des notes, le directeur et les scénaristes. Les datasets ayant tous les deux une colonne `genre`,
j'ai fusionné les deux colonnes pour avoir une liste de valeur unique. Par exemple si un film a pour genre_x [`Action`]
et pour genre_y [`Adventure`, `Action`], le film aura pour genre [`Action`, `Adventure`].

## Feature Engineering
J'ai fait le feature engineering principalement lors de la creation du dataset.

J'ai combiné la colonne `directors` et `writers` pour avoir une colonne `cast` qui contient la liste combinée des deux
colonnes avec les id uniques et dans l'ordre.

J'ai converti la colonne `rating` en float.

J'ai supprimé la colonne `tconst` car cela faisait doublon avec la colonne `movieId`.

J'ai crée la matrice user-item interaction pour l'utilisé plus tard dans le collaborative filtering.

## Model Development

Mon model est un modèle de recommandation basé sur le filtrage collaboratif. J'ai utilisé
la méthode de Singular Value Decomposition (SVD) pour prédire les ratings des films pour chaque utilisateur. J'ai utilisé
la fonction `svds` de la librairie `scipy.sparse.linalg` pour faire la décomposition de la matrice user-item interaction car 
cette fonction est plus rapide que `SVD` de `numpy.linalg`.

## Recommendation Algorithm
J'ai voulu implémenter un modèle de content-based filtering à l'aide de la méthode cosine similarity
mais cette methode ne pouvait pas fonctionner car le dataset était trop grand donc il n'y avait pas assez de ressource pour
le calcule. J'ai donc décidé de faire des k-means clustering sur ce que j'ai appélé les metadata c'est à dire le genre du film,
le cast, l'année du film. J'ai utilisé `CountVectorizer` pour transformer les metadata en vecteur et ensuite j'ai utilisé
`KMeans` pour faire le clustering. J'ai ajouté chaque id de cluster dans le dataset.

Pour combiné les deux modèles de collaboratif filtering et de content-based filtering et crée un modèle de prédiction
hybride, j'ai d'abort pris la prédiction donnée par SVD puis j'ai utilisé les clusters générés par le Kmeans pour ajuster
les prédictions en fonction de la similarité entre les metadata.

Pour avoir un résultat entre 1 et 5, j'ai normalisé les prédictions en utilisant la fonction `MinMaxScaler`.

## Evaluation
Pour evalué le modèle, j'ai séparé le dataset en train et test avec 80/20 de proportion. J'ai ensuite formé des couple de users
sur le dataset de test. Le temps de calcul aurait été trop long avoir j'ai pris 10 couples de manière aléatoire. Ensuite,
j'ai calculé la ratings combiné de chaque couple pour chaque film que le couple a en commun. J'ai calculé la prédictions
pour ce film pour le couple. J'ai mis le tout dans des listes et j'ai calculé la root mean squared error entre les deux listes.

## Conclusion
D'après la root mean square error qui est de 1.69, le modèle a une prédiction assez bonne mais pas complétement. Il faudrait
peut-être améliorer le modèle en changeant par example la méthode de clustering ou le nombre de cluster. On pourrait aussi
encodé les métadata différement dans le modèle de content-based filtering, j'ai pensé a utilisé la méthode de `LabelEncoder`
pour encoder les genres et les cast au lieu d'utilisé la methode de construire une string avec les metadata et de faire le
`CountVectorizer`. Ma méthode pour combiné les users était de prendre la moyenne des ratings des deux users,
cela peut aussi être amélioré en utilisant une autre méthode.
