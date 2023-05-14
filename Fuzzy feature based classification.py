# Projekt na przedmiot PRN 

import numpy as np 
import pandas as pd 
from fuzzywuzzy import fuzz
import time
from subprocess import check_output
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay

print(check_output(["ls", "../input"]).decode("utf8"))

# Poniżej podajemy ścieżkę do dataset
data_frame = pd.read_csv('../input/train21/train1.csv')

# Analiza pytań
def extract_features(df):
    #Procentowe podobieństwo pytań
    df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    #Procentowe podobieństwo podzbiorów
    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    #Procentowe podobieństwo tokenizowanego ciągu znaków po eliminacji zdublowanych podzbiorów
    df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    #Procentowe podobieństwo tokenizowanego i posortowanego podzbioru
    df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    #Procentowe podobieństwo tokenizowanego ciągu znaków po usunięciu zdublowanych słów
    df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    #Procentowe podobieństwo tokenizowanego ciągu znaków
    df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    return df
# Przypisanie analizowanych wartości do DataFrame'u
df = extract_features(data_frame)
print (df)

# Usuwamy zbędne kolumny
feature_columns = df.columns.drop(['id','question1', 'question2', 'is_duplicate'])

X_normalized = normalize(df[feature_columns], norm='l2',axis=1, copy=True, return_norm=False)
print (X_normalized)
x_train, x_test, y_train, y_test = train_test_split(X_normalized, df['is_duplicate'],random_state=1,test_size=0.2)

result_cols = ["Classifier", "Accuracy", "Build Time", "Use Time", "Macro Precision", "Micro Precision", "Macro Sensitivity", "Micro Sensitivity", "F1 Macro", "F1 Micro"]
result_frame = pd.DataFrame(columns=result_cols)

# Poniżej znajdują się wybrane do testów klasyfikatory
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GaussianNB()]


for clf in classifiers:
    name = clf.__class__.__name__
   
    # Mierzymy czas trenowania klasyfikatora
    t_start = time.time()
    clf.fit(x_train, y_train)
    t_end = time.time()
    t_time_ms_sk = (t_end - t_start) * 1000
    
    # Mierzymy czas predykcji 
    p_start = time.time()
    predicted = clf.predict(x_test)
    p_end = time.time()
    p_time_ms_sk = ((p_end - p_start) * 1000)
    
    # Mierzymy parametry działania danego klasyfikatora 
    acc = metrics.accuracy_score(y_test,predicted)
    
    # Dokładność przewidywania
    acc_sk = accuracy_score(y_test, predicted),
    
    # Precyzja makro
    prec_macro_sk = precision_score(y_test, predicted, average='macro', zero_division=0),
    
    # Precyzja mikro
    prec_micro_sk = precision_score(y_test, predicted, average='micro', zero_division=0),
    
    # Czułość makro
    rec_macro_sk = recall_score(y_test, predicted, average='macro', zero_division=0),
    
    # Czułość mikro
    rec_micro_sk = recall_score(y_test, predicted, average='micro', zero_division=0),
    
    # F1 makro
    f1_macro_sk = f1_score(y_test, predicted, average='macro', zero_division=0),
    
    # F1 mikro
    f1_micro_sk = f1_score(y_test, predicted, average='micro', zero_division=0)
    
    acc_field = pd.DataFrame([[name, acc*100, t_time_ms_sk, p_time_ms_sk, list(prec_macro_sk)[0]*100, list(prec_micro_sk)[0]*100, list(rec_macro_sk)[0]*100, list(rec_micro_sk)[0]*100, list(f1_macro_sk)[0]*100, f1_micro_sk*100]], columns=result_cols)
    result_frame = result_frame.append(acc_field)

    #  Tworzymy Confusion Matrix
    cm_sk = confusion_matrix(y_test, predicted)
    disp_sk = ConfusionMatrixDisplay(confusion_matrix=cm_sk)
    # disp_sk("124as")
    
    disp_sk.plot()
    print(clf)
    
    
    print("| Dokładność       |", round(list(acc_sk)[0]*100 ,3),"%")
    print("| Czas budowy [ms] |", round(t_time_ms_sk,3))
    print("| Czas użycia [ms] |", round(p_time_ms_sk,3))
    print("| Precyzja makro   |", round(list(prec_macro_sk)[0]*100 ,3) ,"%")
    print("| Precyzja mikro   |", round(list(prec_micro_sk)[0]*100, 3) ,"%")
    print("| Czułość makro    |", round(list(rec_macro_sk)[0]*100, 3),"%")
    print("| Czułość mikro    |", round(list(rec_micro_sk)[0]*100, 3) ,"%")
    print("| F1 makro         |", round(list(f1_macro_sk)[0]*100, 3) ,"%")
    print("| F1 mikro         |", round(f1_micro_sk*100, 3) ,"%")
   
def make_plot(dataFrame,XGraph, postfix):
    
    plt.show()
    plot = sns.barplot(x=XGraph, y='Classifier', data=dataFrame, color="r")
    
    for p in plot.patches:
        width = p.get_width()    # get bar length
        plot.text(width + 1,       # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
                '{:1.2f}'.format(width), # set variable to display, 2 decimals
                ha = 'left',   # horizontal alignment
                va = 'center')  # vertical alignment
    
    plt.xlabel( XGraph + ' ' + postfix)
    plt.title('Classifier '+ XGraph)
    return 0

make_plot(result_frame, 'Accuracy', '%')
make_plot(result_frame, 'Build Time', 'ms')
make_plot(result_frame, 'Use Time', 'ms')
make_plot(result_frame, 'Macro Precision', '%')
make_plot(result_frame, 'Micro Precision', '%')
make_plot(result_frame, 'Macro Sensitivity', '%')
make_plot(result_frame, 'Micro Sensitivity', '%')
make_plot(result_frame, 'F1 Macro', '%')
make_plot(result_frame, 'F1 Micro', '%')
