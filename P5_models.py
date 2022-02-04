from typing import cast
from enum import Enum
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from yellowbrick.cluster import KElbowVisualizer
from kneed import KneeLocator
import mlflow


class Algorithm(Enum):
    KMeans = 'kmeans'
    MiniBatchKmeans = 'minikmeans'
    DBSCAN = 'dbscan'


class Vectorizer(Enum):
    Count = 'count'
    TfIdf = 'tf-idf'


def preprocessing(dataset: pd.DataFrame, vectorizer: Vectorizer):
    """ Add features engineering to the dataset """
    max_features = 3000
    mlflow.log_param('max_features', max_features)
    mlflow.log_param('vectorizer', vectorizer.value)

    # CountVectorizer
    if vectorizer == Vectorizer.Count:
        count_vectorizer = CountVectorizer(lowercase=False,
                                           max_features=max_features)
        count_matrix = count_vectorizer.fit_transform(dataset['Body'])
        df = pd.DataFrame(count_matrix.toarray(), index=dataset.index,
                          columns=count_vectorizer.get_feature_names_out())
        df.reset_index(inplace=True, drop=True)

    # Tf-Idf
    elif vectorizer == Vectorizer.TfIdf:
        tf_vectorizer = TfidfVectorizer(lowercase=False,
                                        max_features=max_features)
        tf_matrix = tf_vectorizer.fit_transform(dataset['Body'])
        df = pd.DataFrame(tf_matrix.toarray(), index=dataset.index,
                          columns=tf_vectorizer.get_feature_names_out())
        df.reset_index(inplace=True, drop=True)

    n_components = 500
    mlflow.log_param('n_components', n_components)
    pca = PCA(n_components=n_components)
    df_projected = pca.fit_transform(df)
    mlflow.log_metric('explained variance',
                      pca.explained_variance_ratio_.cumsum()[-1] * 100)
    return (df_projected, df)


def make_clusters(X: npt.NDArray, algorithm: Algorithm):
    if algorithm == Algorithm.MiniBatchKmeans:
        kelbow_viz = KElbowVisualizer(MiniBatchKMeans(random_state=5),
                                      k=(16, 24))
        kelbow_viz.fit(X)
        mlflow.log_figure(kelbow_viz.fig, './artifacts/kelbow.png')
        mlflow.log_metric('kelbow', cast(float, kelbow_viz.elbow_value_))
        plt.close()

        kmeans = MiniBatchKMeans(kelbow_viz.elbow_value_, random_state=5)
        kmeans.fit(X)
        labels = pd.Series(kmeans.labels_, name='cluster-label')
        value_dict = dict(labels.value_counts())
        value_counts = {str(k): int(v) for k, v in value_dict.items()}
        mlflow.log_dict(value_counts, './artifacts/cluster-sizes.json')
        return labels

    elif algorithm == Algorithm.DBSCAN:
        X_sample = X[0:50000]
        nb_neighbors = 10
        nearest_neighbors = NearestNeighbors(n_neighbors=nb_neighbors)
        nearest_neighbors.fit(X_sample)
        distances, _ = nearest_neighbors.kneighbors(X_sample)

# Get max distance between neighbors
        max_distances = np.sort(distances[:, nb_neighbors - 1])

# Find an elbow
        index = np.arange(len(max_distances))
        knee = KneeLocator(index, max_distances, curve='convex',
                           direction='increasing', interp_method='polynomial')
        knee.plot_knee(figsize=(10, 10))
        plt.xlabel("Points")
        plt.ylabel("Distance")
        mlflow.log_figure(plt.gcf(), './artifacts/dbelbow.png')
        mlflow.log_metric('dbelbow', cast(float, knee.elbow_y))
        dbscan = DBSCAN(min_samples=100, eps=knee.elbow_y)
        dbscan.fit(X_sample)
        mlflow.log_metric('db cluster', len(dbscan.labels_))


def display_topics(model, feature_names, no_top_words):
    html_out = ['<ul>']
    for topic_idx, topic in enumerate(model.components_):
        html_out.append(f'<li>Topic {topic_idx}: ')
        html_out.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        html_out.append('</li>')
    html_out.append('</ul>')
    return html_out


if __name__ == "__main__":
    with mlflow.start_run():
        dataset = cast(pd.DataFrame, pd.read_csv('./cleaned_dataset.csv',
                       converters={'Tags': lambda x: eval(x)}))
        for vectorizer in Vectorizer:
            with mlflow.start_run(run_name=vectorizer.value, nested=True):
                (X, df) = preprocessing(dataset, vectorizer)

                # Clustering
                labels = make_clusters(X, Algorithm.MiniBatchKmeans)

                dataset_labels = df.assign(cluster_label=labels,
                                           tags=dataset['Tags'])
                cluster_sum = cast(pd.DataFrame, dataset_labels
                                   .groupby('cluster_label').sum())
                html_out = ['<ul>']
                for label in cluster_sum.index:
                    top10_topics = cluster_sum.loc[label]\
                            .sort_values(ascending=False)\
                            .head(10).index.to_list()
                    html_out.append((
                        f'<li>Topics: '
                        f'{top10_topics}'
                        f'</li>'))
                    labels_tag = set([tag for tags in
                                     dataset_labels[
                                         dataset_labels['cluster_label'] == label
                                         ]['tags']
                                     .to_list()
                                     for tag in tags])
                    html_out.append(f'<li>Tags: {labels_tag}</li>')
                html_out.append('</ul>')
                mlflow.log_text(''.join(html_out), 'topics-per-clusters.html')
                decomp_model = None

                # LDA / NMF
                no_topics = 20
                if vectorizer == Vectorizer.Count:
                    mlflow.log_param('no_topics', no_topics)
                    decomp_model = LatentDirichletAllocation(n_components=no_topics,
                                                             max_iter=5,
                                                             learning_method='online',
                                                             learning_offset=50.,
                                                             random_state=0).fit(df)
                else:
                    decomp_model = NMF(n_components=no_topics, random_state=1,
                                       alpha_W=.1, l1_ratio=.5,
                                       init='nndsvd').fit(df)
                topics_out = display_topics(decomp_model, df.columns, 10)
                mlflow.log_text(''.join(topics_out), 'LDA-topics.html')
                # TODO log lda.score() (log-likelihood as score)
                # TODO timer la prédiction de chaque model

                # Supervised approach
                # pip = Pipeline()
                X_train, X_test, y_train, y_test = train_test_split(X, dataset['Tags'], test_size=0.2, random_state=34)
                y_train = MultiLabelBinarizer().fit_transform(y_train)
                y_test = MultiLabelBinarizer().fit_transform(y_test)
                ovr = OneVsRestClassifier(LogisticRegression(n_jobs=-1), n_jobs=-1)
                ovr.fit(X_train, y_train)
                ovr_score = ovr.score(X_test, y_test)
                mlflow.log_metric('OvR score', ovr_score)

                ovo = OneVsOneClassifier(SVC(), n_jobs=-1)
                ovo.fit(X_train, y_train)
                ovo_score = ovo.score(X_test, y_test)
                mlflow.log_metric('OvO score', ovo_score)
