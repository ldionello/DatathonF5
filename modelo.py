import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def carregar_base(caminho):
    df = pd.read_csv(caminho)
    for col in ['keywords_encontradas', 'keywords_vaga']:
        df[col] = df[col].apply(lambda x: x if pd.notnull(x) else '')
    return df


def treinar_modelo(df, target_col='target'):
    features_text = ['keywords_encontradas', 'keywords_vaga']
    features_cat = ['nivel_academico', 'nivel_ingles', 'nivel_espanhol',
                    'area_atuacao', 'tipo_contratacao', 'nivel_profissional']

    # Garantir que não existam NaNs nas colunas categóricas e textuais
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

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            max_depth=5,
            scale_pos_weight=3,
            subsample=0.8
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


def prever_candidato(modelo, dados_input):
    return modelo.predict(dados_input), modelo.predict_proba(dados_input)[:, 1]
