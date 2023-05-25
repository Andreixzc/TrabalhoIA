import pickle
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import scipy.stats as st
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import NearMiss
import xgboost
import matplotlib.pyplot as plt

with open('raw1.pkl', 'rb') as f:
    dataset = pickle.load(f)


def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))



def k_fold_cross_validation_com_grid_e_under_completo(k, model, dataset):
    X_train = np.concatenate((dataset[0], dataset[1]), axis=0)  # Características de treino
    y_train = np.concatenate((dataset[2], dataset[3]), axis=0)  # Rótulos de treino
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]
    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados
    scores = []  # Lista para armazenar as métricas de cada fold
    # params = {
    #     'criterion':  ['gini', 'entropy'],
    #     'max_depth':  [None, 2, 4, 6, 8, 10],
    #     'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    # }
    #XGB
    params = { 
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampler nos dados de treino do fold
        nearmiss = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train_fold,y_train_fold)

        # GridSearch no fold
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados
        best_params = grid_search.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros
        model.set_params(**best_params)
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        #Positivo e negativo.
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])

        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])

        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        # Relatório de classificação do fold
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:", len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("--------------------------------------------")
   
    # # Exibição das médias e intervalos de confiança
    print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))


def k_fold_cross_validation_com_grid_e_over_completo(k, model, dataset):
    X_train = np.concatenate((dataset[0], dataset[1]), axis=0)  # Características de treino
    y_train = np.concatenate((dataset[2], dataset[3]), axis=0)  # Rótulos de treino
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]
    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados
    scores = []  # Lista para armazenar as métricas de cada fold
    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Oversampler nos dados de treino do fold
        ros = RandomOverSampler(random_state=0)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_fold, y_train_fold)

        # GridSearch no fold
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados
        best_params = grid_search.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros
        model.set_params(**best_params)
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        #Positivo e negativo.
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])

        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])

        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        # Relatório de classificação do fold
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:", len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("--------------------------------------------")
   
    # # Exibição das médias e intervalos de confiança
    print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))

def k_fold_cross_validation_com_grid_e_over_parcial(k, model, dataset):
    X_train = dataset[0]  # Características de treino
    X_test = dataset[1]  # Características de teste
    y_train = dataset[2]  # Rótulos de treino
    y_test = dataset[3]  # Rótulos de teste
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]

    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados

    scores = []  # Lista para armazenar as métricas de cada fold

    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Oversampler nos dados de treino do fold
        ros = RandomOverSampler(random_state=0)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_fold, y_train_fold)

        # GridSearch no fold
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados
        best_params = grid_search.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros
        model.set_params(**best_params)
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])
        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])
        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:",len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("------------------------------")

    # Avaliação final no conjunto de teste
    y_pred_test = model.predict(X_test)
    final_score = model.score(X_test, y_test)
    scores.append(final_score)
    # Exibição das médias e intervalos de confiança
    # # Exibição das médias e intervalos de confiança
    print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))
    # Relatório de classificação do conjunto de teste
    print("Relatorio Final do conjunto inteiro de dados (sem k-fold):")
    print(classification_report(y_test, y_pred_test))
    print("------------------------------")

def k_fold_cross_validation_com_grid_e_under_parcial(k, model, dataset):
    X_train = dataset[0]  # Características de treino
    X_test = dataset[1]  # Características de teste
    y_train = dataset[2]  # Rótulos de treino
    y_test = dataset[3]  # Rótulos de teste
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]

    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados

    scores = []  # Lista para armazenar as métricas de cada fold

    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }



    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampler nos dados de treino do fold
        nearmiss = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train_fold,y_train_fold)

        # GridSearch no fold
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados
        best_params = grid_search.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros
        model.set_params(**best_params)
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])
        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])
        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:",len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("------------------------------")

    # Avaliação final no conjunto de teste
    y_pred_test = model.predict(X_test)
    final_score = model.score(X_test, y_test)
    scores.append(final_score)
    # Exibição das médias e intervalos de confiança
    # # Exibição das médias e intervalos de confiança
    print(recall_inferior)
    # print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    # print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))
    # Relatório de classificação do conjunto de teste
    print("Relatorio Final do conjunto inteiro de dados (sem k-fold):")
    print(classification_report(y_test, y_pred_test))
    print("------------------------------")



def k_grafico(k, model, dataset):
    X_train = np.concatenate((dataset[0], dataset[1]), axis=0)  # Características de treino
    y_train = np.concatenate((dataset[2], dataset[3]), axis=0)  # Rótulos de treino
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]
    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados
    scores = []  # Lista para armazenar as métricas de cada fold
    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampler nos dados de treino do fold
        nearmiss = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train_fold,y_train_fold)

        # GridSearch no fold
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados
        best_params = grid_search.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros
        model.set_params(**best_params)
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        #Positivo e negativo.
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])

        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])

        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        # Relatório de classificação do fold
        print(f"Fold {fold + 1}:")
        print("Melhores parametros para o fold:", fold + 1)
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("--------------------------------------------")

    # Plotando gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = {
        'F1 Score Superior': f1_score_superior,
        'F1 Score Inferior': f1_score_inferior,
        'Recall Superior': recall_superior,
        'Recall Inferior': recall_inferior,
        'Precisão Superior': precisao_superior,
        'Precisão Inferior': precisao_inferior
    }

    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        ax = axes[i // 3, i % 3]
        ax.plot(range(1, k + 1), metric_values, marker='o')
        ax.set_title(metric_name)
        ax.set_xlabel('Fold')
        ax.set_ylabel('Valor')

    plt.tight_layout()
    plt.show()

    # Exibição das médias e intervalos de confiança
    print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: ", np.round(fscore_valid / k, 2))
    print("Recall médio na validação das classes Inferior e Superior: ", np.round(recall_valid / k, 2))
    print("Precisão média na validação das classes Inferior e Superior: ", np.round(precision_valid / k, 2))



import matplotlib.pyplot as plt

def k_fold_cross_validation_com_grid_e_under_completo(k, model1, model2, dataset):
    X_train = np.concatenate((dataset[0], dataset[1]), axis=0)  # Características de treino
    y_train = np.concatenate((dataset[2], dataset[3]), axis=0)  # Rótulos de treino
    f1_score_superior_1 = []
    f1_score_inferior_1 = []
    recall_superior_1 = []
    recall_inferior_1 = []
    precisao_superior_1 = []
    precisao_inferior_1 = []
    f1_score_superior_2 = []
    f1_score_inferior_2 = []
    recall_superior_2 = []
    recall_inferior_2 = []
    precisao_superior_2 = []
    precisao_inferior_2 = []
    precision_valid_1 = [0, 0]
    recall_valid_1 = [0, 0]
    fscore_valid_1 = [0, 0]
    precision_valid_2 = [0, 0]
    recall_valid_2 = [0, 0]
    fscore_valid_2 = [0, 0]
    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados
    scores_1 = []  # Lista para armazenar as métricas de cada fold (Modelo 1)
    scores_2 = []  # Lista para armazenar as métricas de cada fold (Modelo 2)
    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampler nos dados de treino do fold (Modelo 1)
        nearmiss = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train_fold, y_train_fold)

        # GridSearch no fold (Modelo 1)
        grid_search_1 = GridSearchCV(estimator=model1, param_grid=params, scoring='accuracy', cv=3)
        grid_search_1.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados (Modelo 1)
        best_params_1 = grid_search_1.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros (Modelo 1)
        model1.set_params(**best_params_1)
        model1.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold (Modelo 1)
        y_pred_1 = model1.predict(X_val_fold)
        score_1 = model1.score(X_val_fold, y_val_fold)
        scores_1.append(score_1)
        # Positivo e negativo.
        precision_1, recall_1, fscore_1, support_1 = precision_recall_fscore_support(y_val_fold, y_pred_1, average=None)
        precision_valid_1 = np.add(precision_valid_1, precision_1)
        recall_valid_1 = np.add(recall_valid_1, recall_1)
        fscore_valid_1 = np.add(fscore_valid_1, fscore_1)
        f1_score_superior_1.append(fscore_1[0])
        f1_score_inferior_1.append(fscore_1[1])
        recall_superior_1.append(recall_1[0])
        recall_inferior_1.append(recall_1[1])
        precisao_superior_1.append(precision_1[0])
        precisao_inferior_1.append(precision_1[1])

        # Undersampler nos dados de treino do fold (Modelo 2)
        nearmiss = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train_fold, y_train_fold)

        # GridSearch no fold (Modelo 2)
        grid_search_2 = GridSearchCV(estimator=model2, param_grid=params, scoring='accuracy', cv=3)
        grid_search_2.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados (Modelo 2)
        best_params_2 = grid_search_2.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros (Modelo 2)
        model2.set_params(**best_params_2)
        model2.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold (Modelo 2)
        y_pred_2 = model2.predict(X_val_fold)
        score_2 = model2.score(X_val_fold, y_val_fold)
        scores_2.append(score_2)
        # Positivo e negativo.
        precision_2, recall_2, fscore_2, support_2 = precision_recall_fscore_support(y_val_fold, y_pred_2, average=None)
        precision_valid_2 = np.add(precision_valid_2, precision_2)
        recall_valid_2 = np.add(recall_valid_2, recall_2)
        fscore_valid_2 = np.add(fscore_valid_2, fscore_2)
        f1_score_superior_2.append(fscore_2[0])
        f1_score_inferior_2.append(fscore_2[1])
        recall_superior_2.append(recall_2[0])
        recall_inferior_2.append(recall_2[1])
        precisao_superior_2.append(precision_2[0])
        precisao_inferior_2.append(precision_2[1])

        # Relatório de classificação do fold (Modelo 1)
        print(f"Fold {fold+1} - Modelo 1:")
        print("Melhores parametros para o fold:")
        print(best_params_1)
        print(classification_report(y_val_fold, y_pred_1))
        print("--------------------------------------------")

        # Relatório de classificação do fold (Modelo 2)
        print(f"Fold {fold+1} - Modelo 2:")
        print("Melhores parametros para o fold:")
        print(best_params_2)
        print(classification_report(y_val_fold, y_pred_2))
        print("--------------------------------------------")

    # Média das métricas de desempenho (Modelo 1)
    avg_precision_1 = precision_valid_1 / k
    avg_recall_1 = recall_valid_1 / k
    avg_fscore_1 = fscore_valid_1 / k

    # Média das métricas de desempenho (Modelo 2)
    avg_precision_2 = precision_valid_2 / k
    avg_recall_2 = recall_valid_2 / k
    avg_fscore_2 = fscore_valid_2 / k

    # Gráfico comparativo das métricas de desempenho (Modelos 1 e 2)
    fig, ax = plt.subplots()
    index = np.arange(2)
    bar_width = 0.35
    opacity = 0.8

    rects1 = ax.bar(index, [avg_precision_1[0], avg_recall_1[0]], bar_width, alpha=opacity, color='b', label='Modelo 1')
    rects2 = ax.bar(index + bar_width, [avg_precision_2[0], avg_recall_2[0]], bar_width, alpha=opacity, color='g', label='Modelo 2')

    ax.set_xlabel('Métricas')
    ax.set_ylabel('Valores')
    ax.set_title('Comparação das métricas de desempenho (Positivo)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Precisão', 'Revocação'])
    ax.legend()

    fig.tight_layout()
    plt.savefig('comparacao_positivo.png')

    fig, ax = plt.subplots()
    index = np.arange(2)
    bar_width = 0.35
    opacity = 0.8

    rects1 = ax.bar(index, [avg_precision_1[1], avg_recall_1[1]], bar_width, alpha=opacity, color='b', label='Modelo 1')
    rects2 = ax.bar(index + bar_width, [avg_precision_2[1], avg_recall_2[1]], bar_width, alpha=opacity, color='g', label='Modelo 2')

    ax.set_xlabel('Métricas')
    ax.set_ylabel('Valores')
    ax.set_title('Comparação das métricas de desempenho (Negativo)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Precisão', 'Revocação'])
    ax.legend()

    fig.tight_layout()
    plt.savefig('comparacao_negativo.png')

    # Gráfico comparativo das métricas de desempenho F1-Score (Modelos 1 e 2)
    fig, ax = plt.subplots()
    index = np.arange(2)
    bar_width = 0.35
    opacity = 0.8

    rects1 = ax.bar(index, [avg_fscore_1[0], avg_fscore_1[1]], bar_width, alpha=opacity, color='b', label='Modelo 1')
    rects2 = ax.bar(index + bar_width, [avg_fscore_2[0], avg_fscore_2[1]], bar_width, alpha=opacity, color='g', label='Modelo 2')

    ax.set_xlabel('Classes')
    ax.set_ylabel('F1-Score')
    ax.set_title('Comparação do F1-Score entre as classes')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Positivo', 'Negativo'])
    ax.legend()

    fig.tight_layout()
    plt.savefig('comparacao_f1_score.png')

    # Retorno dos scores finais (Modelos 1 e 2)
    return scores_1, scores_2



rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
xgb_model = xgboost.XGBClassifier()

# print("Rf com over:")
# k_fold_cross_validation_com_grid_e_over(5,rf,dataset)
# print("Rf com under:")
# k_fold_cross_validation_com_grid_e_under(5,rf,dataset)

print("DT")
# k_fold_cross_validation_com_grid_e_under_parcial(5,dt,dataset)
# k_fold_cross_validation_com_grid_e_over_completo(5,dt,dataset)
# k_fold_cross_validation_com_grid_e_under_completo(5,dt,dataset)
# k_fold_cross_validation_com_grid_e_under_completo(5,xgb_model,dataset)
# k_grafico(5,dt,dataset)
# k_grafico(5,rf,dataset)
k_fold_cross_validation_com_grid_e_under_completo(5,dt,rf,dataset)
print("rf")
# k_fold_cross_validation_com_grid_e_under_parcial(5,rf,dataset)
# k_fold_cross_validation_com_grid_e_over_completo(5,rf,dataset)


# print(interval_confidence(recall))
