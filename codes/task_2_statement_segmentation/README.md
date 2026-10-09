# Task 2 sentence-segmentation dependencies

`requirements.txt` pins spaCy and its imported Click runtime dependency. The
Spanish pipeline is a separate model package and must be installed into the
existing project environment:

```bash
ig_env/bin/python -m spacy download es_core_news_md
```

For a reproducible spaCy 3.8 installation, the equivalent model wheel is:

```bash
ig_env/bin/python -m pip install \
  https://github.com/explosion/spacy-models/releases/download/es_core_news_md-3.8.0/es_core_news_md-3.8.0-py3-none-any.whl
```

Do not substitute an English pipeline. The segmentation script stops with an
installation message if the requested Spanish model cannot be loaded.
