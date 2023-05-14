<h1>Porównanie algorytmów klasyfikacji na zbiorze rozmytym</hi1>


<h2>Wstęp</h2>

W tym projekcie głównym celem było dokonanie porównania różnych parametrów związanych z klasyfikacją zbiorów rozmytych. Do jego realizacji wykorzystano zbiór danych “Quora Question Pairs”, język Python oraz bibliotek fuzzywuzzy, seaborn i scikit-learn itp.

**Opis zbioru danych**

Zbiorem danych jest zbiór 50 000 rekordów zawierających dwa pytania. Zbiór został stworzony w celu konkursu mającego na celu zidentyfikowanie, czy dwa pytania mają te same znaczenie.

Informacje o atrybutach:
* id - numer identyfikacyjny pary pytań
* qid1, qid2 - unikalne numery identyfikacyjne każdego z pytań, dostępne tylko w zbiorze train.csv
* question1, question 2 - pytania
* is_duplicate - wyznacznik jednoznaczności pytań. Jeżeli równy 1 - pytania mają to samo znaczenie.


<h2>Pakiet fuzzywuzzy</h2>

Pakiet “fuzzywuzzy” używany jest do porównywania tekstu z użyciem odległości Levenshteina do kalkulacji różnicy pomiędzy sekwencjami tekstu.  Odległością Levenshteina nazywamy najmniejszą liczbą działań prostych, które przetransformują jeden napis na drugi.
Przykładem jest np. wyraz “orczyk” i “oracz”. Odległość między tymi napisami wynosi 3, ponieważ wymaga dwóch działań usunięcia liter “y’” oraz “k” i wstawienia litery “a”. Przy identycznych napisach odległość wynosi 0.
Funkcja “extract_features” zajmuje się analizą pytań i wpisaniem wartości do dataframe’u. 
Wywołuje ona pojedyncze funkcje zawarte w pakiecie fuzzywuzzy.

**fuzz.QRatio**

Funkcja QRatio kalkuluje odległość Levenstheina i zwraca procentową wartość podobieństwa dwóch wyrażeń. W przypadku, gdy dwa wyrażenia są identyczne, funkcja zwraca wartość 100%. 

**fuzz.partial_ratio**

Partial_ratio bierze pod uwagę podzbiory ciągów znaków, które porównuje, a następnie zwraca współczynnik podobieństwa. Na przykład porównując “Stoi na stacji lokomotywa” z “lokomotywa”, algorytm zwróci wartość 100%.

**fuzz.token_sort_ratio**

Metody Token ignorują wielkość liter i interpunkcję. token_sort_ratio sortuje ciągi znaków alfanumeryczne i stosuje funkcję fuzz.Ratio. Dzięki temu kolejność słów w porównywanych ciągach znaków nie ma znaczenia.

**fuzz.token_set_ratio**

Funkcja ta działa podobnie jak token_sort_ratio. Ignoruje ona jednak zdublowane słowa.

**fuzz.partial_token_sort_ratio**

Powyższa funkcja działa podobnie do token_sort_ratio. Po tokenizacji i sortowaniu używa jednak funkcji partial_ratio zamiast QRatio.

**fuzz.partial_token_set_ratio**

Funkcja po tokenizacji i eliminacji zdublowanych wyrazów używa funkcji partial_ratio zamiast QRatio.


<h2>Algorytmy klasyfikacji</h2>

W projekcie użyliśmy następujących algorytmów klasyfikacji z pakietu scikit-learn
* KNeighborsClassifier
* DecisionTreeClassifier
* RandomForestClassifier
* AdaBoostClassifier
* GaussianNB

**KNeighborsClassifier**

Jest to algorytm klasyfikacji oparty na metodzie k-najbliższych sąsiadów (k-NN bądź k-Nearest Neighbors). K-NN to metoda uczenia nadzorowanego. Przydziela nowy punkt danych do klasy na podstawie etykiet klasy najbliższych punktów szkoleniowych. Klasa nowego punktu jest określana przez większość głosów k najbliższych punktów. Im większa liczba k, tym bardziej wygładzona decyzja klasyfikacji. W projekcie algorytm został użyty z parametrem k = 3.

**DecisionTreeClassifier**

Klasyfikator ten służy do tworzenia modeli klasyfikacyjnych na podstawie drzew decyzyjnych. Drzewem nazywamy strukturę składającą się z węzłów reprezentujących testy na atrybutach oraz krawędzie reprezentujące wyniki testów. Głównym celem jest podział danych na bardziej jednorodne grupy poprzez zastosowanie odpowiednich testów na poszczególnych atrybutach. W tym celu drzewo decyzyjne budowane jest w sposób iteracyjny poprzez wybieranie kolejnych atrybutów i ich testów tak, aby jak najlepiej podzielić dane.

**RandomForestClassifier**

RandomForestClassifier oparty jest na algorytmie drzew decyzyjnych. Łączy on wiele drzew decyzyjnych w las. Każde drzewo trenowane jest na podstawie losowo wybranych próbek z danych treningowych. W przypadku klasyfikacji wynik pozyskiwany jest poprzez głosowanie większościowe drzew. Las losowy jest skuteczny w redukcji overfittingu oraz w przypadku gdy dane zawierają wiele funkcji, niektóre są nieistotne bądź silnie skorelowane.

**AdaBoostClassifier**

Klasyfikator ten jest klasyfikatorem zespołowym działającym poprzez łączenie wielu prostych (słabych) klasyfikatorów aby utworzyć jeden silny klasyfikator. Uczy on się poprzez iteracyjne trenowanie słabych klasyfikatorów na kolejno wagach zaktualizowanych w zależności od wyniku poprzedniej iteracji. Wagi te używane są do ważenia wkładu każdego słabego klasyfikatora w końcową decyzję klasyfikacyjną. Zaimplementowany jest z użyciem algorytmu SAMME (Stagewise Additive Modelling using a Multiclass Exponential loss function)


**GaussianNB**

Algorytm ten to algorytm klasyfikacji Bayesowski z użyciem rozkładu Gaussa. Jest on prosty i skuteczny. Zakłada, że każda cecha wejściowa jest niezależna od innych cech. Tworzy on model wykorzystujący dane uczące i prawdopodobieństwo każdej klasy, a następnie przewiduje klasę nowych danych na podstawie maksymalnego prawdopodobieństwa a posteriori. Model ten jest szczególnie skuteczny w przypadku danych, w których każda cecha jest ciągła, a klasy są rozdzielone w sposób zbliżony do rozkładu Gaussa.
