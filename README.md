# Data Mining Project (SS2025)

Link zum GitHub-Repository: https://github.com/Imran17192/DataMiningProject2025

## Übersicht

Dieses Projekt enthält die Daten und Skripte für den Projektteil des Moduls Data Mining im Sommersemester 2025.

## Nutzung

Dieses Projekt wird mit JetBrains' IDE PyCharm entwickelt. Alles was getan werden muss, um durch den Code zu stöbern, ist es, die IDE zu installieren und das Projekt zu importieren. Fehlende Packages können durch Ausführung des Befehls `python3 -m pip install -r requirements.txt` installiert werden. Um die Datenanalyse zu starten genügt es, das Skript main.py mit dem Befehl `python3 main.py` auszuführen.

## Projektstruktur

### main.py

Die Datei main.py enthält den Haupt-Programmcode. Die einzelnen Schritte des Data Mining werden dabei durch jeweilige Funktionen (dm_part\[1,2,3\]) durchgeführt

### paths.py

Die Datei paths.py enthält einige Konstanten, welche die Projektstruktur repräsentieren, sowie eine Funktion, welche für einen gegebenen Pfad eine Liste aller darin enthaltenen json-Dateien zurückgibt.

### data

Der data-Ordner enthält vorgegebenen Daten für das Data Mining.

### plots

Der plots-Ordner enthält die bei der bzw. für die Datenanalyse generierten Plots.

### predictions

Der predictions-Ordner enthält die bei der Klassifikation generierten Predictions.

### scripts

Der scripts-Ordner ist modular aufgebaut und enthält für verschiedene Aspekte des Data Mining entsprechende Unterordner, welche py-Dateien enthalten, die von der Datei main.py importiert werden und die für das Data Mining benötigten Funktionen implementieren.

## Vorbereitung

Um die benötigten Packages zu installieren den Befehl `python3 -m pip install -r requirements.txt` ausführen.

## Ausführung
Ausführung des Codes über den Befehl `python3 main.py`.
