# Scan2CAD 1.0.1 – geführte Rückführung von Scan-Netzen (STL) in STEP

Amon P. Lanz · Apache License 2.0 (siehe `LICENSE`, `NOTICE`) · Zitierhinweis in `CITATION.cff` / `CITATION.bib` und im Programm unter *Über / Zitieren* (rechte obere Ecke). Ein großer Teil des Codes wurde mit Claude Opus 5.5 (Anthropic) geschrieben.

Python-Programm (PySide6 + VTK/pyvista + OpenCASCADE/OCP), das Dreiecksnetze aus 3D-Scans
(z. B. Creality Raptor) in echte CAD-Geometrie übersetzt und als **geschlossenen STEP-Volumenkörper**
exportiert. Oberfläche Deutsch / English (Menü *Sprache / Language* oder Auswahl in der
Werkzeugleiste; das Programm startet neu und nimmt den aktuellen Stand mit).

## Neu in 1.0.1

* **Boden schließen** robust bei Einrissen/Schlitzen im Rand (werden vorher geschlossen), selbst-
  schneidende Bodenpolygone werden erkannt und anders trianguliert (Ohr-Abschneiden statt Delaunay).
* **Bohrungsumgebung als Drehprofil**: Ringnuten (O-Ring), Einstiche, Fasen um Bohrungen werden aus dem
  gemeinsamen Profil erkannt und exakt als Rotationskörper abgezogen; Schein-Bohrungen (Nutwände) und
  schief gefittete Achsen verschwinden. Auch per Knopf im Bohrungen-Menü.
* **Bohrungen**: fast zylindrische Kegel zählen als Bohrwand (vorher übersehene Bohrung), konkave
  Kantenrundungen in Innenecken sind keine Ø0,8-„Bohrungen“ mehr (Umfangsprüfung).
* **Kantenrundungen und Splitter**: Scan-Verrundungen der Kanten werden scharfe Kanten (Radius bleibt
  vermerkt, beim Export optional echte Verrundung), winzige Restflächen fallen weg.
* **Drehteile** (Trichter, Buchsen) komplett aus dem (r, z)-Profil; **CAD-Exporte** als Eingabe
  (grob facettierte STL) werden als solche erkannt und exakt nachgebaut.
* **Schräg angeschnittene Bohrungen/Kanäle**: das Werkzeug wird an offenen Enden so weit verlängert, dass kein Keil stehen bleibt (z. B. halb angeschnittener Kanal, der schräg aus einer Wand läuft).
* **STEP-Aufbau**: plausibilitätsgeprüfte Kanten, Rettung über Freiform-/Facettenbereiche, so dass auch
  stark verrundete Teile geschlossen werden.
* **Abgleich Features ↔ STEP** bei jeder Vorschau und jedem Export: welches Feature exakt im Körper
  steckt, welches ersetzt wurde oder fehlt, und welche Scanbereiche der Körper nicht trifft; Vorschau
  nach Herkunft der Flächen oder nach Abweichung zum Scan eingefärbt.
* **Oberfläche flüssiger**: FXAA statt SSAA, Feature-Darstellung wird nach Rechenschritten nicht mehr
  komplett neu tesselliert, sehr feine Scans werden gleich zu Beginn ausgedünnt.

## Schnellstart (Windows)

1. ZIP in einen Ordner entpacken (Pfad bitte ohne `!`).
2. `start.bat` doppelklicken.
   * **Erster Start:** legt `.venv` im Programmordner an und installiert die Abhängigkeiten
     (einmalig ca. 400 MB).
   * **Danach:** eine vorhandene `.venv` mit passender `requirements.txt` wird nur noch gestartet.
     1.0.1 braucht gegenüber früheren Ständen ein kleines zusätzliches Paket (`mapbox_earcut`, Triangulierung von
     Böden/Deckeln) – `start.bat` installiert es beim ersten Start automatisch nach.
   * STL/OBJ/PLY per Drag & Drop auf `start.bat` → wird direkt geöffnet (`.s2c` = Projekt).
   * `start.bat --reinstall` erzwingt eine Neuinstallation, `start.bat --lang=en` startet englisch.
3. Voraussetzung: 64-bit Python 3.11–3.14 (getestet mit 3.13).
4. Zum Ausprobieren: `beispiel/testteil.stl` → Werkzeugleiste **✨ Auf gut Glück**.
   Sollmaße: Grundkörper 60 × 40 × 15 mm, Ecken R4; Zapfen Ø16, 8 mm hoch, Kante R2;
   Kuppel = Kugel R9 (Mitte 6 mm unter der Oberseite); Durchgangsbohrung Ø6 mit 90°-Senkung auf Ø10.
5. Zweites Beispiel mit **offenem Boden** (wie vom Drehteller): `beispiel/flansch_offen.stl`, beliebig
   gedreht im Raum. Sollmaße: Platte 60 × 60 × 10 mm, Ecken R6; Bund Ø30 bis Höhe 22 mm;
   Mittelbohrung Ø12 durch; 4 × Ø6,6 durch auf Lochkreis Ø44; 2 × Kernloch Ø4,2 (M5) 8 mm tief
   auf ±9,5 mm. Der Scanner sieht die Bohrungen nicht bis zum Grund – Tiefe/Boden legt man im
   Bohrungen-Menü fest, die Plattendicke per Bedingung *Abstand Oberseite Platte – Boden = 10 mm*.

Fehler beim Start landen im Konsolenfenster und in `scan2cad.log`.

## ✨ Auf gut Glück (Werkzeugleiste / rechter Reiter)

Fährt den ganzen Ablauf automatisch ab und zeigt nach **jedem Schritt** das Zwischenergebnis
(„nach jedem Schritt anhalten“ ist voreingestellt). In der Pause kann man beliebig eingreifen –
Seiten links benutzen, Features löschen oder neu fitten, Füllungen ändern, Bohrungen korrigieren,
`Strg+Z` – und dann mit **▶ Weiter** fortfahren; der nächste Schritt arbeitet mit dem aktuellen
Stand. Schritte lassen sich abhaken, „⏭ Ein Schritt“ führt genau einen aus. Jeder Schritt ist ein
eigener Rückgängig-Punkt.

1. Netz reparieren (Teilstücke, Orientierung, nicht-mannigfaltige Kanten)
2. Rauschen messen → alle Toleranzen daraus setzen
3. kleine Löcher glatt füllen (bis Ø = 3 % der Teilgröße)
4. offenen Boden schließen (nur wenn ein großer, flacher Rand da ist) und restliche Löcher füllen
5. vereinfachen (nur über 150 000 Dreiecken)
6. grob ausrichten (Boden nach unten bzw. Hauptachsen mit größter ebener Fläche bzw. offenem Rand
   unten, Mitte XY, Unterseite auf z = 0)
7. Formen erkennen (RANSAC) → 8. ähnliche zusammenführen
9. **Ausrichtung aus den Features optimieren** + Achsen/Normalen exakt auf X/Y/Z stellen
   (liegen die Konstruktionsflächen schräg, wird zuerst so gedreht, dass möglichst viel Fläche
   achsparallel ist)
10. **Drehteil?** Ist fast die ganze Oberfläche rotationssymmetrisch (Trichter, Buchse, Welle), werden
    alle Flächen aus dem gemeinsamen (r, z)-Profil bestimmt – exakt koaxial, Rundungen als Tori
11. Bohrungen erkennen + glatte Werte
12. **Bohrungsumgebung als Drehprofil** (s. u.): Um eine Bohrung liegen oft weitere
    rotationssymmetrische Elemente – Ringnut (O-Ring), Einstich, Fase. Einzeln gefittet entstehen
    daraus schiefe Zylinderstücke und Schein-Bohrungen (die Nutwände sehen wie kurze Bohrungen aus).
    Stattdessen: Achse = Normale der Eintrittsfläche, Lage aus den Flächennormalen (alle Normalen
    einer Rotationsfläche schneiden die Achse), dann das gemeinsame (r, z)-Profil der Umgebung in
    Strecken und Kreise zerlegen. Was rundum gemessen ist, zählt; Schein-Bohrungen fallen weg, die
    echte Bohrung bekommt Achse und Ø aus dem Profil, jede übrige Luftzone unter der Fläche wird ein
    abgezogener **Rotationskörper** (Polygon mit Eckenradien, Maße auf 0,1/0,05 mm geglättet), eine
    reine Fase an der Bohrung wird zur Kegelsenkung. Die Kantenverrundung des Scans an den Öffnungen
    wird dabei scharf.

13. **Kantenrundungen und Splitter**: schmale Rundungsstreifen der Scan-Kanten werden scharfe Kanten
    (Radius bleibt vermerkt, beim Export optional als echte Verrundung), winzige Restflächen fallen weg
14. Symmetrien suchen → 15. Restflächen → Freiform
16. **STEP-Vorschau** im Modus Scan-Topologie mit Dichtheits-Prüfung

**CAD-Exporte als Eingabe** (grob facettierte STL aus einem CAD-Programm) werden erkannt: dann
zählen die exakten Eckpunkte statt der Dreiecksmitten, Toleranzen kommen aus der Facettierung
(Sehnenfehler) statt aus dem Rauschen, und der Boden wird nicht „geschlossen".

## Seiten (links)

| Seite | Inhalt |
|---|---|
| **1 Scan** | *Prüfen*: Rauschen σ → Toleranzen, Löcher, Topologie. *Reparieren*, *Mannigfaltig machen*, *Ausreißer-Spitzen glätten*. *Löcher & Boden*: einzeln (Liste / `H` + Klick) oder bis Ø füllen (glatt / straff / eben), **offenen Boden schließen**. *Füllungen nachbearbeiten*. *Korrigieren (Auswahl)*: glätten, fairen, auf Feature projizieren, löschen + neu füllen. *Vereinfachen* nach Toleranz. |
| **2 Ausrichten** | Schnell (Hauptachsen, 90°, Boden, Mitte), **Optimieren** (Fangwinkel, danach Achsen gerade stellen, Boden auf z = 0), 3-2-1 aus Auswahl / Feature / **Symmetrie** / Weltachse, numerisch, Originallage. |
| **3 Erkennen** | RANSAC (Ebene, Zylinder, Kugel, Kegel, Torus), **ähnliche zusammenführen**, **Bohrungen erkennen**, Restflächen → Freiform. |
| **4 Modellieren** | Pinsel/Füllen → Typ → Fitten → Erweitern → Übernehmen (Primitive, Freiform, Extrusion, Rotation). |
| **5 Bohrungen** | Liste aller Bohrungen, ◀ ▶ blättern, Editor (s. u.), Bohrung aus Auswahl oder per `K`-Klick. |
| **6 Symmetrie & Bedingungen** | Symmetrien suchen/anzeigen/löschen/anlegen/erzwingen; Bedingungen definieren, vorschlagen lassen und gemeinsam gegen den Scan lösen. |
| **7 Export** | **Scan-Topologie** (Standard, dichter Volumenkörper), *Getrimmte Flächen*, *Körper/Booleans*; STEP-Vorschau (`P`), STEP, Netz, facettiertes STEP, Bericht. |

Rechts: **Features** (Liste, Rolle, Reihenfolge, Neu-Fitten, Einrasten), **Messen**,
**✨ Auf gut Glück**.

## Export „Scan-Topologie“ – wie der geschlossene STEP entsteht

Statt die Flächen gegeneinander zu verschneiden (instabil, besonders an tangentialen Übergängen),
wird die **Nachbarschaft der Flächen aus dem Scan** genommen (ähnlich „Exact Surfacing"):

1. Jedes Dreieck gehört genau einer Trägerfläche (den gefitteten Features). Freie Dreiecke
   bekommen den nächstgelegenen Träger; die Grenzen werden nach Abstand *und* Normale geglättet,
   Inseln und „Einschnürungen“ entfernt. Gleichwertige Träger (z. B. zwei getrennt erkannte
   Hälften einer Bohrung) werden zusammengelegt. Bereiche, die kein Träger erklärt (z. B. die
   Membran eines nicht erfassten Bohrungsgrunds), bekommen eine **Hilfsebene**.
2. Rundum-Flächen (Zylinder, Kegel, Torus, Kugel) werden an einer gemeinsamen Ebene geteilt,
   die Naht liegt immer gegenüber der Fläche.
3. Kanten: exakter Schnitt der beiden Träger (Newton), erkannt als **Gerade/Kreis**, sonst
   B-Spline; an **tangentialen Übergängen** (Rundungen) wird die Berührkurve bestimmt, die Ecken
   werden auf diese Kurve gesetzt. Ecken liegen auf allen angrenzenden Trägern (Gauß-Newton).
4. Flächen mit **gemeinsam genutzten Kanten** → Schale → Volumenkörper (`BRepCheck`).
5. Körper-Features (Bohrungen, Ringnuten, Zapfen …): ihre Dreiecke werden vorher durch ebene Deckel
   ersetzt (ringförmige Bereiche als Fläche mit Loch), danach wird die exakte Geometrie per Boolean
   angebracht.
6. Kleine Dellen/Fehlstellen mitten in einer Ebene, die keine Form erklärt, werden eben geschlossen.
   **Rettung**, wenn die Hülle so nicht dicht wird (verrundete Freiform-Teile, Spielzeug, Gehäuse):
   Bereiche, deren Träger nicht zur Nachbarschaft passt (Ecken ohne gemeinsamen Punkt, Kanten weit
   neben der Scan-Grenze, ungültige Flächen), werden als **B-Spline-Fläche durch ihre Randkanten**
   neu aufgebaut; die Kanten laufen dort entlang der geglätteten Scan-Grenze, auf die gültige
   Nachbarfläche gelegt. Vorher wird versucht, die Feature-Flächen zu erhalten: wo die Schnittkante
   zweier Flächen nicht zum Scan passt (echte Verrundung, die kein Feature ist), wird der Streifen
   dazwischen als eigener **Freiform-Übergang** gebaut und die beiden Flächen bleiben exakt. Was dann noch offen ist, wird **facettiert** (Dreiecke, deren Rand exakt auf
   den Nachbarkanten liegt), Restspalte werden gefüllt und alles vernäht. Der Bericht nennt, wie
   viele Bereiche Freiform bzw. facettiert sind; ein Körper, den die Formprüfung bemängelt, wird als
   „geschlossen, aber mit Fehlern" gemeldet. Abziehen/Vereinigen von Körper-Features wird verworfen,
   wenn das Volumen danach nicht plausibel ist.
7. Optional (*Optionen Scan-Topologie*): erkannte **Kantenrundungen verrunden** ab einem
   Mindestradius – die Kanten zwischen den beiden Ebenen bekommen eine echte Verrundung.

Die Vorschau meldet „✔ Geschlossener Volumenkörper" oder zeigt offene Kanten **rot**.

**Wichtig – der STEP ist nicht die Summe der sichtbaren Features.** Er wird aus der Scan-Topologie
gebaut; die Features liefern nur die Trägerflächen. Abweichungen entstehen dort, wo Dreiecke keinem
Feature gehören (sie gehen an den nächsten Träger), wo die Rettung eine Ersatz-Freiform oder
Dreiecke einsetzt, und wo ein Boolean unplausibel ist und ausgelassen wird. Deshalb prüft jede
Vorschau und jeder Export **jedes Feature gegen den fertigen Körper** (*Abgleich Features ↔ STEP*
im Protokoll):

* je Feature: *exakt* (Körper liegt auf dem Feature, gleicher Flächentyp), *als Freiform*, *als
  Dreiecke*, *anderer Flächentyp*, *nur teilweise* oder *fehlt*;
* **Scanbereiche ohne Entsprechung** im Körper (z. B. nicht erkannte kleine Bohrungen, die der
  Nachbarträger überdeckt) mit Lage, Fläche und Abstand – in der Vorschau **gelb** markiert;
* Vorschau-Farben (*Vorschau färben*): **Herkunft** (grau = exakt auf dem Feature, orange =
  Ersatz-Freiform, rot = Scan-Dreiecke), **Abweichung zum Scan** oder einfarbig.

Die Statuszeile unter der Vorschau fasst das zusammen („18/18 Features exakt im Körper“).
Test: Aus einem realistischen Scan (Marching Cubes, Rauschen σ = 0,015 mm, nicht erfasste
Bohrungstiefen, Fehlstellen) entsteht so ein Volumenkörper, dessen Volumen < 0,2 % vom
CAD-Original abweicht.

## Bohrungen & Gewinde (Seite 5, Werkzeug `K`)

* **Erkennen:** automatisch (konkave Zylinder, koaxiale Teile werden zusammengefasst) oder `K` +
  Klick auf die Bohrungswand bzw. *Aus Auswahl*. Achse und Ø kommen nur aus dem sauber erfassten
  Teil der Wand; Eintrittspunkt aus der umgebenden Fläche; Kegelsenkung/Flachsenkung und
  Bohrergrund werden mit erkannt. Alles im Bohrungsvolumen (Murks, Füllmembranen) gehört zur
  Bohrung und wird beim Export durch die exakte Bohrung ersetzt.
* **Boden nicht erfasst:** dann gilt die sichtbare Tiefe, die Bohrung ist gelb markiert – Tiefe und
  Boden hier festlegen (ein Scanner kann eine Durchgangsbohrung, die nur von oben sichtbar war,
  nicht von einem Sackloch unterscheiden). Endet die Wand an einem offenen Scan-Rand, wird
  „Grundloch, Boden nicht erfasst" angenommen; liegt jenseits der Lücke ein koaxiales Wandstück
  gleichen Ø (von der Gegenseite gescannt), wird die Bohrung als durchgehend erkannt.
* **Klick `K`:** wächst vom angeklickten Dreieck nur entlang des daraus geschätzten Zylinders –
  läuft also nicht über verrundete Kanten auf die Nachbarflächen; Klicks auf konvexe Flächen
  (z. B. Außenseite eines Bunds) werden abgewiesen.
* **Editor:** Eintrittspunkt, Richtung (X/Y/Z, umkehren), Ø, Tiefe, **Boden** flach /
  Bohrerspitze (Winkel) / Kugel / durchgehend; **Senkung** keine / Kegelsenkung (Ø, Winkel) /
  Flachsenkung (Ø, Tiefe); **Gewinde** metrisch (Regel-/Feingewinde, Steigung, Länge),
  Darstellung *nur Angabe* (Kernloch), *Nenn-Ø* oder *modelliert* (experimentell: facettierte
  Schraubenlinie, große Dateien).
* **Werte glätten:** Ø auf Normwerte (Kernloch, Durchgangsloch fein/mittel/grob, Bohrer),
  Winkel auf 90/118/120/…, Achse auf X/Y/Z bzw. vorhandene Ebenen-Normalen, Tiefe runden.
  „Gewinde vorschlagen" erkennt aus dem Ø das passende Kernloch.
* **Umgebung als Drehprofil** (auch automatisch im Ablauf): Um eine Bohrung liegen oft weitere
  rotationssymmetrische Elemente – Ringnut (O-Ring), Einstich, Fase. Einzeln gefittet entstehen
  daraus schiefe Zylinderstücke und Schein-Bohrungen (die Nutwände sehen wie kurze Bohrungen aus).
  Stattdessen: Achse = Normale der Eintrittsfläche, Lage aus den Flächennormalen (alle Normalen
  einer Rotationsfläche schneiden die Achse), dann das gemeinsame (r, z)-Profil der Umgebung in
  Strecken und Kreise zerlegen. Was rundum gemessen ist, zählt; Schein-Bohrungen fallen weg, die
  echte Bohrung bekommt Achse und Ø aus dem Profil, jede übrige Luftzone unter der Fläche wird ein
  abgezogener **Rotationskörper** (Polygon mit Eckenradien, Maße auf 0,1/0,05 mm geglättet), eine
  reine Fase an der Bohrung wird zur Kegelsenkung. Die Kantenverrundung des Scans an den Öffnungen
  wird dabei scharf.

## Messen (Werkzeug `M`)

Zwei Stellen anklicken (oder zwei Features wählen). Liegt die Stelle auf einem Feature, wird dessen
exakte Geometrie benutzt, sonst eine lokal gefittete Ebene bzw. ein Zylinder (Umkreis einstellbar).
Ausgabe je nach Paar: **Winkel** Fläche–Fläche / Achse–Achse / Achse–Fläche, Abstände (parallel,
windschief, Punkt–Ebene, Punkt–Achse), Ø.

## Symmetrien

* **Suchen** (ganzes Teil oder nur Auswahl): Spiegelebenen, **Rotationskörper (∞)** und
  **n-zählige Drehung (360°/n)**. Kandidaten aus Hauptachsen und Feature-Richtungen, Verfeinerung
  per Punkt-zu-Ebene-ICP, Güte = Anteil der Scanpunkte, deren Bild wieder auf dem Scan liegt
  (Schwelle einstellbar). Gilt eine höhere Zähligkeit nur näherungsweise (z. B. 4-zählig bis auf
  zwei Gewindelöcher), wird zusätzlich die exakte (2-zählig) gemeldet.
* **Muster aus Features:** Lochkreise / Kreismuster, Spiegelpaare, koaxiale Gruppen – mehrere
  Achsen und Ebenen gleichzeitig.
* Liste mit Anzeige und **Löschen**, manuell **anlegen** (aus Ebene/Mittelebene/Achse),
  **Erzwingen** (legt „invariant unter Symmetrie"-Bedingungen an), Ausrichten an einer Symmetrie.

## Bedingungen – rückwirkend über den Scan erzwungen

Typen: parallel, senkrecht, Winkel, koaxial, koplanar, Abstand, Ø, Radius, gleiche Radien,
Achse in Ebene, symmetrisch zu Ebene, invariant unter Symmetrie, fixieren.
Bezüge: Features, **Mittelebenen zweier Ebenen**, Symmetrien, Weltachsen/-ebenen. Beispiele:

* „Fläche A ∥ Fläche B" → *parallel*.
* „4-zählige Achse geht durch die Mittelebenen zweier Flächenpaare" → Bezüge
  *Mitte A1 | A2* und *Mitte B1 | B2*, zweimal *Achse in Ebene*.
* „Achse parallel, aber nicht koinzident zu einer anderen Achse" → *parallel* (statt *koaxial*).

**Lösen** fittet alle beteiligten Features gemeinsam neu: jede Fläche bleibt so nah wie möglich an
ihren echten Scanpunkten, die Bedingungen werden als stark gewichtete Terme erfüllt; Flächen ohne
Messdaten (festgelegter Boden, erzwungene Füllebenen, Symmetrieachsen, Bohrungen) folgen nur den
Bedingungen. Der Bericht zeigt Abweichung vorher/nachher und die RMS jeder Fläche. **Vorschlagen**
listet fast erfüllte Beziehungen (fast parallel, fast koaxial, Ø nahe Normwert …) zum Abhaken.

## Offener Boden

Scans vom Drehteller sind unten oft offen. *Seite 1 → Löcher & Boden → Boden schließen:*

* Die Wände am offenen Rand werden **tangential bis zu einer Ebene verlängert** und eben gedeckelt.
  Ebene: Ausgleichsebene des Rands, äußerster Randpunkt, z = Wert, Abstand zu einer Bezugsebene oder
  genau eine Bezugsebene. Weitere Ränder nahe der Ebene (Bohrungen) werden mit geschlossen.
  Würde die Verlängerung bei sehr verrauschtem Rand ein sich selbst schneidendes Bodenpolygon
  ergeben, werden die Wandrichtungen stärker geglättet bzw. der Rand senkrecht projiziert (Meldung).
* Es entsteht die Ebene „Boden" mit dem Kennzeichen **festgelegt**: ohne Messdaten, also nie aus
  dem Scan gefittet, sondern über Bedingungen bestimmt – z. B. *Abstand Deckfläche–Boden = 10 mm*.
  Beim Lösen wandert der Deckel im Netz mit. *Boden-Ebene ändern* wählt die Ebene neu.
* Jede andere Füllung lässt sich per **Als Ebene erzwingen** genauso zur festgelegten Ebene machen
  (Ausgleichsebene des Rands oder koplanar zu einem Feature).

## Füllungen nachbearbeiten

Alle Füllungen werden protokolliert (Liste auf Seite 1) und bleiben bearbeitbar: **neu füllen**
(glatt / straff / eben), **fairen** (Falten und Riffel glätten), **als Ebene erzwingen**,
**entfernen** (Loch wieder offen), auswählen. Künstliche Dreiecke sind hellviolett, zählen beim
Fitten nur mit Gewicht 0,1 und dienen RANSAC nie als Startpunkt – Riffel einer Membranfüllung
erzeugen daher keine eigenen Ebenen. Vorhandene Splitter-Ebenen: *Seite 3 → ähnliche zusammenführen*.

## Weitere Scan-Werkzeuge

* **Rauschen schätzen** (lokale Quadrik-Fits, robust gegen Krümmung) → Toleranzen.
* **Ausreißer-Spitzen glätten** (Punkte > k·σ über der lokalen Fläche).
* **Auf Feature projizieren**: Scan-Dreiecke exakt auf eine gefittete Fläche ziehen.
* **Löschen + neu füllen**: Artefakte (Stützen, Klebereste, Reflexionsmüll) ausschneiden und das Loch
  glatt schließen; **Fairen** für Beulen/Falten.
* **Mannigfaltig machen**: doppelte und gefaltete Dreiecke, Kanten mit > 2 Dreiecken, gekippte
  Normalen.

## Tasten

`N`/`Esc` Navigieren · `B` Pinsel · `E` Radierer · `F` Füllen · `H` Lochrand · `K` Bohrung ·
`M` Messen · `[` `]` Pinselgröße · `C` Auswahl leeren · `Leertaste` Fitten · `G` Erweitern ·
`Enter` Übernehmen · `P` STEP-Vorschau · `R` Kamera · `Strg+Z`/`Strg+Y` Rückgängig/Wiederholen ·
`Strg+S` Projekt · `Strg+E` STEP · `F1` Hilfe. Mal-Modi: links malen, Strg+links radieren,
rechts/Alt+links drehen; ein Grafiktablett wirkt wie die Maus.

Rückgängig/Wiederholen umfasst Auswahl, Features, Netzänderungen, Füllungen, Ausrichtung,
Symmetrien und Bedingungen. Das Projekt (`.s2c`) speichert all das inkl. Füllungen.

## Bekannte Grenzen

* **Kaputte Netze:** Bei stark zerrissenen Netzen kann die Scan-Topologie keine dichte Hülle bauen;
  die Vorschau zeigt die offenen Kanten rot. Abhilfe: *Prüfen*, *Mannigfaltig machen*, Löcher füllen,
  notfalls Modus *Körper/Booleans*.
* **Tangentiale Übergänge** zwischen Freiformflächen werden als B-Spline genähert (Kantentoleranz im
  STEP entsprechend größer); bei Ebenen/Zylindern/Tori sind es exakte Geraden/Kreise.
* **Kugelflächen**, die mehr als eine Halbkugel umfassen, werden geteilt; sehr große Kugelanteile mit
  Polen können Fehler melden.
* **Modellierte Gewinde** sind experimentell (facettiert, große Dateien) – für die Fertigung reicht
  „nur Angabe" + Gewindebezeichnung.
* Was der Scan nicht sieht (Bohrungsgrund, wahre Lage eines offenen Bodens), wird angenommen und
  markiert – die Entscheidung liegt beim Anwender (Bohrungen-Menü, Bedingungen).
* **Stark verrundete Freiform-Teile** (Spielzeug, Konsumgüter): Die Erkennung findet Ebenen/Zylinder,
  die Rundungen dazwischen werden zunächst Kanten, dann Freiform-Übergänge; wo das topologisch nicht
  aufgeht, entstehen Freiform- bzw. facettierte Bereiche (siehe Rettung). Das Ergebnis ist
  geschlossen und folgt dem Scan (Volumen typ. < 1–2 %), ist aber kein „sauberes" CAD-Modell. Beispiel
  Wasserpistole (Testscan): geschlossen, Volumen −0,1 %, Oberfläche ≈ 31 % exakt, 20 % Freiform,
  49 % Scan-Dreiecke; die Vorschau dauert dort mehrere Minuten. Der *Abgleich Features ↔ STEP* zeigt
  genau, welche Flächen betroffen sind – für Konstruktionsänderungen die wichtigen Flächen gezielt
  auf Seite 4 nachmodellieren.
* **Extrusionen** werden nicht automatisch als Profil × Höhe erkannt; ein prismatischer Bereich
  entsteht aber auch so exakt aus Ebenen/Zylindern. Für ein bearbeitbares Profil: Seite 4,
  Mantelflächen anmalen, Typ *Extrusion*.
* Entwickelt und getestet unter Linux (Xvfb) mit Python 3.13, OCP 8.0.1 (OCCT 8), VTK 9.7.1,
  pyvista 0.49.1, PySide6 6.12; unter Windows über `start.bat`.

## Tests

```
python tests/make_scan_like.py scan.stl 0.25 0.015 [--part=flansch] [--open-bottom]   # realistischer Scan
python tests/test_core.py [ordner]                    # ganzer Ablauf ohne GUI, Volumen vs. CAD-Original
xvfb-run -a python tests/gui_smoke.py scan.stl ordner [--en]    # GUI: Auf gut Glück, Bohrungen,
                                                      # Bedingungen, Messen, Export, Projekt + Screenshots
xvfb-run -a python tests/gui_tools.py scan_offen.stl ordner     # GUI: Einzelwerkzeuge (Boden, Füllungen,
                                                      # Bohrung per Klick, Drehprofil, Symmetrie, Export-Modi)
python tests/test_real.py eigener_scan.stl ... [--step]   # eigene Scans: ganzer Ablauf, Zusammenfassung,
                                                      # geschlossen?, Volumen Körper vs. Netz, optional STEP
```

## Ordner

```
app/main.py               Einstieg (Sprache, Projekt-/Netz-Argument)
app/core/                 Geometrie-Kern (ohne GUI nutzbar)
    autopilot.py          Schritte von „Auf gut Glück"
    document.py           Projektzustand (Netz, Features, Symmetrien, Bedingungen, Lage)
    brep_seg.py           Scan-Topologie → B-Rep (Regionen, Kanten, Ecken, Volumenkörper)
    carriers.py           Trägerflächen (Projektion, Abstand, OCC-Fläche)
    bores.py              Bohrungen: Erkennung, Parameter, Normtabellen, OCC-Aufbau
    symmetry.py           Spiegel-, Dreh-, Rotationssymmetrie, Muster
    constraints.py        Bedingungen + gemeinsamer Neu-Fit (Least Squares)
    analysis.py           Messen, Ausrichtung optimieren, Zusammenführen
    mesh_ops.py           Netz-Werkzeuge, Rauschen, Mannigfaltigkeit, Boden schließen, Füllungen
    holes.py              Löcher finden/füllen (eben, straff, glatt)
    export.py             Export-Modi, Verstopfen der Körper-Features, STEP
    i18n.py               Deutsch/English
    …                     fitting, ransac, features, freeform, sweeps, trim, occ_build, project …
app/gui/                  Oberfläche: main_window.py, page_*.py (eine Datei je Seite),
                          panel_*.py (Features, Messen, Auf gut Glück), viewer.py, workers.py
tests/                    Scan-Generator, Kern- und GUI-Test
```

---

### English summary

Scan2CAD turns 3D-scan meshes into CAD geometry and exports a **closed STEP solid**. Start
`start.bat` (creates `.venv` on first run; `--lang=en` for English, or switch in the toolbar).
**✨ Feeling lucky** runs repair → noise/tolerances → hole filling → open-base closing → simplify →
alignment → shape detection → merge → alignment optimisation → turned-part detection → hole
detection → hole surroundings as turned profile (ring grooves, undercuts, chamfers) → edge blends
and fragments → symmetries → freeform → STEP preview, pausing after each step so you can inspect
and intervene.
The **scan-topology** export derives face adjacency from the scan, computes exact
intersection/tangency edges and builds a watertight solid; holes (with countersinks, threads,
drill points) are applied by booleans. Constraints (parallel, distance, coaxial, axis-in-mid-plane,
symmetry …) are enforced by a joint refit against the scan; surfaces without scan data (an
extrapolated floor, forced fill planes, symmetry axes) are driven by constraints only.
