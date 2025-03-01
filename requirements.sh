# curl -sS https://bootstrap.pypa.io/get-pip.py | python3
pip install --upgrade pip
pip install --upgrade --ignore-installed setuptools  #fix https://github.com/tensorflow/tensorflow/issues/622
pip install --upgrade scikit-learn tqdm nltk editdistance joblib graphviz pandas gensim matplotlib bokeh optuna
pip install --upgrade tensorflow
pip install --upgrade keras
pip install --upgrade torch tf-keras accelerate transformers