# projects

1. LogReg.py (kod) & Projekt regresji logistycznej.py

    Regresja logistyczna – przewidywanie zwycięzcy meczu tenisowego na podstawie danych pośrednich zebranych w latach 2000-2020

Cel
    Celem badania było oszacowanie dokładności modelu regresji logistycznej mającego na celu wskazanie zwycięzcy meczu tenisowego na podstawie historycznych danych pośrednich  (warunki kortu, średnia kursu bukmacherskiego, ranking gracza, liczba rozegranych setów). Jeden wiersz danych reprezentuje jeden rozegrany mecz. Na potrzeby modelu wybrane zostały dane tylko dla jednego z graczy - Diego Schwartzman.

Załadowanie danych
    Zmienna zależna data została rozdzielona na 2 grupy - miesiąc, rok. Zmienne jakościowe, przy pomocy funkcji pakietu pandas 'get.dummies' zostały przekształcone na zmienne ilościowe. Dane zostały ograniczone dla wyników dla jednego z graczy, w ten sposób mogliśmy precyzyjnie określić funkcję celu - y jako zwycięstwo lub przegrana tego gracza. 
Pełny opis danych znajduje się na stronie: http://tennis-data.co.uk/notes.txt
