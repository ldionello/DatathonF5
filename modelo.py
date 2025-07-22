import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def carregar_base(caminho):
    df = pd.read_csv(caminho)
    for col in ['keywords_encontradas', 'keywords_vaga']:
        df[col] = df[col].fillna('')
    return df


def treinar_modelo(df, target_col='target'):
    features_text = ['keywords_encontradas', 'keywords_vaga']
    features_cat = ['nivel_academico', 'nivel_ingles', 'nivel_espanhol',
                    'area_atuacao', 'tipo_contratacao', 'nivel_profissional']

    for col in features_text + features_cat:
        df[col] = df[col].fillna('')

    X = df[features_text + features_cat]
    y = df[target_col]

    text_transformer = Pipeline([('vect', CountVectorizer(max_features=1000))])
    cat_transformer = Pipeline(
        [('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('text_encontradas', text_transformer, 'keywords_encontradas'),
        ('text_vaga', text_transformer, 'keywords_vaga'),
        ('cat', cat_transformer, features_cat)
    ])

    # Pré-processa os dados antes do SMOTE
    X_processed = preprocessor.fit_transform(X)

    # Aplica o SMOTE sobre os dados numéricos já transformados
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        max_depth=5,
        scale_pos_weight=3,
        subsample=0.8
    )

    model.fit(X_resampled, y_resampled)

    return model, preprocessor  # Retorna também o preprocessor para uso posterior


def prever_candidato(modelo, preprocessor, dados_input):
    X_input = preprocessor.transform(dados_input)
    return modelo.predict(X_input), modelo.predict_proba(X_input)[:, 1]
