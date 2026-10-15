<img src="Vogel_Orange.svg" width="64" alt="Amon P. Lanz’s bird logo">

# Scan2CAD

Turn 3D scans (STL / OBJ / PLY) into **closed STEP solids with design intent**. Work through it guided, step by step, or let **✨ Feeling lucky** do the whole run and step in wherever you like.

**Version 1.0.1 · Windows (Python) · Apache-2.0 · Amon P. Lanz**

**[Download Scan2CAD_V1.0.1.zip](https://raw.githubusercontent.com/AmLanz/Scan2CAD/main/Scan2CAD_V1.0.1.zip)** · [Source code](Scan2CAD_V1.0.1/Scan2CAD/) · [Report an issue](https://github.com/AmLanz/Scan2CAD/issues) · [Deutsch ↓](#deutsch)

<!-- Screenshot: ![Scan2CAD](screenshot.png) -->

## What it does

- **Checks and repairs the scan.** It estimates the noise, which sets all tolerances, then fills holes, closes an open base (turntable scans) and thins out very dense meshes.
- **Aligns the part to its design axes.** Planes and axes are snapped exactly onto X / Y / Z.
- **Detects shapes:** planes, cylinders, cones, spheres and tori, with duplicates merged and B-spline freeforms for the rest. Turned parts (funnels, bushings, shafts) are fitted from one shared (r, z) profile, so they come out exactly coaxial.
- **Handles holes.** Diameters snap to standard values. It recognizes countersinks and counterbores and drill-point bottoms, and flags diameters that match a tap drill as a possible thread. O-ring grooves and chamfers around holes are fitted as a turned profile. Where the scanner did not see the bottom, you decide the depth.
- **Uses symmetries and constraints** (distances, parallel / perpendicular, equal radii …) and solves them together against the scan.
- **Exports a closed solid** from the scan topology. Faces lie on the fitted features, edges are their exact intersection curves, and holes and grooves are cut by boolean operations.
- **Checks features against the STEP** at every preview and export. It reports which feature made it in exactly, which one was replaced (freeform or triangles) or is missing, and which scan areas the solid does not cover. The preview can be coloured by face origin or by deviation from the scan.
- German / English interface, projects (`.s2c`), undo / redo.

## Installation (Windows)

Requires **64-bit Python 3.11–3.14** from [python.org](https://www.python.org/downloads/) (tested with 3.13).

1. Download **Scan2CAD_V1.0.1.zip** and extract it to a permanent folder (path without `!`).
2. Double-click **`start.bat`**. The first start creates a `.venv` inside the folder and installs the dependencies (about 400 MB, once). After that it just starts.
3. Drag an STL / OBJ / PLY or `.s2c` project onto `start.bat` to open it directly.
   `start.bat --lang=en` starts in English, `start.bat --reinstall` reinstalls the dependencies.

Errors show up in the console window and in `scan2cad.log`.
Other systems: `python -m venv .venv`, then `pip install -r requirements.txt`, then `python app/main.py`.

## Quick start

1. Open `beispiel/testteil.stl` (or your own scan).
2. Click **✨ Feeling lucky** in the toolbar. The autopilot runs all steps (repair → align → detect → holes → symmetry → STEP preview) and pauses after each one. Correct anything on the pages on the left, then press **▶ Continue**.
3. Check the holes on page **5 · Holes** (depth, through / blind, thread) and the symmetries and constraints on page **6**.
4. On page **7 · Export**, open the **STEP preview (P)**, read the feature check in the log, then **Export STEP…**.

A second example with an open base, as it comes off a turntable scanner, is in `beispiel/flansch_offen.stl`. Its target dimensions are listed in the detailed documentation.

## Good to know

- **Prismatic and turned parts** (machined, printed, moulded technical parts) work best. The test parts typically come out with every feature exact, and their volume is 0.05–0.12 % off the original CAD.
- **Heavily rounded freeform parts** (toys, housings) still close as a solid, but some areas become freeform patches or raw scan triangles. The feature check shows exactly where.
- **What the scanner cannot see** (hole bottoms, the true position of an open base) is assumed and flagged. You decide.
- **Not yet:** automatic extrusion recognition (profile × height) and chamfer recognition.
- **Always verify critical dimensions** before manufacturing anything from a scan.

## Documentation

The detailed documentation (in German) is in [Scan2CAD_V1.0.1/Scan2CAD/README.md](Scan2CAD_V1.0.1/Scan2CAD/README.md). It covers all pages and keys, the autopilot steps, the hole tool, how the closed STEP is built, the feature ↔ STEP check and the tests.

## License and citation

Copyright © 2026 **Amon P. Lanz**. Distributed under the **[Apache License 2.0](LICENSE)**; see [NOTICE](Scan2CAD_V1.0.1/Scan2CAD/NOTICE).

Much of the code was written with **Claude Opus 5.5** (Anthropic). The idea, scans, test parts, critique and every decision on what is good enough came from Amon P. Lanz.

Citation is appreciated and optional:

> Amon P. Lanz (2026). *Scan2CAD: Guided Reverse Engineering of 3D Scans to STEP* (version 1.0.1). Computer software.
> https://github.com/AmLanz/Scan2CAD

Use **About / Cite** (top-right corner of the app) to copy a citation or BibTeX entry. [CITATION.cff](Scan2CAD_V1.0.1/Scan2CAD/CITATION.cff) and [CITATION.bib](Scan2CAD_V1.0.1/Scan2CAD/CITATION.bib) are included.

---

<a id="deutsch"></a>

# Scan2CAD (Deutsch)

Macht aus 3D-Scans (STL / OBJ / PLY) **geschlossene STEP-Volumenkörper mit Konstruktionsabsicht**. Man arbeitet sich geführt Schritt für Schritt durch oder lässt **✨ Auf gut Glück** alles durchlaufen und greift ein, wo man will.

**Version 1.0.1 · Windows (Python) · Apache-2.0 · Amon P. Lanz**

**[Scan2CAD_V1.0.1.zip herunterladen](https://raw.githubusercontent.com/AmLanz/Scan2CAD/main/Scan2CAD_V1.0.1.zip)** · [Quellcode](Scan2CAD_V1.0.1/Scan2CAD/) · [Fehler melden](https://github.com/AmLanz/Scan2CAD/issues)

## Was es kann

- **Prüft und repariert den Scan.** Es misst das Rauschen, daraus ergeben sich alle Toleranzen. Dann füllt es Löcher, schließt einen offenen Boden (Drehteller-Scans) und dünnt sehr feine Netze aus.
- **Richtet das Teil an seinen Konstruktionsachsen aus.** Ebenen und Achsen werden exakt auf X / Y / Z gestellt.
- **Erkennt Formen:** Ebenen, Zylinder, Kegel, Kugeln und Tori, Doppelte zusammengelegt, der Rest als B-Spline-Freiform. Drehteile (Trichter, Buchsen, Wellen) entstehen aus einem gemeinsamen (r, z)-Profil und sind dadurch exakt koaxial.
- **Bohrungen:** Durchmesser rasten auf Normwerte ein. Kegel- und Flachsenkungen sowie Bohrerspitzen werden erkannt, Durchmesser, die zu einem Kernloch passen, als mögliches Gewinde markiert. Ringnuten und Fasen um Bohrungen werden als Drehprofil gefittet. Wo der Scanner den Grund nicht gesehen hat, legt man die Tiefe selbst fest.
- **Symmetrien und Bedingungen** (Abstände, parallel / senkrecht, gleiche Radien …) werden gemeinsam gegen den Scan gelöst.
- **Exportiert einen geschlossenen Volumenkörper** aus der Scan-Topologie. Die Flächen liegen auf den gefitteten Features, die Kanten sind ihre exakten Schnittkurven, Bohrungen und Nuten werden per Boolean abgezogen.
- **Abgleich Features ↔ STEP** bei jeder Vorschau und jedem Export. Er zeigt, welches Feature exakt im Körper steckt, welches ersetzt wurde (Freiform oder Dreiecke) oder fehlt, und welche Scanbereiche der Körper nicht trifft. Die Vorschau lässt sich nach Herkunft der Flächen oder nach Abweichung zum Scan einfärben.
- Oberfläche Deutsch / English, Projekte (`.s2c`), Rückgängig / Wiederholen.

## Installation (Windows)

Voraussetzung: **64-bit Python 3.11–3.14** von [python.org](https://www.python.org/downloads/) (getestet mit 3.13).

1. **Scan2CAD_V1.0.1.zip** herunterladen und in einen dauerhaften Ordner entpacken (Pfad ohne `!`).
2. **`start.bat`** doppelklicken. Der erste Start legt im Ordner eine `.venv` an und installiert die Abhängigkeiten (einmalig ca. 400 MB). Danach startet nur noch das Programm.
3. STL / OBJ / PLY oder ein `.s2c`-Projekt per Drag & Drop auf `start.bat` ziehen, dann wird es direkt geöffnet.
   `start.bat --lang=en` startet englisch, `start.bat --reinstall` installiert die Abhängigkeiten neu.

Fehler stehen im Konsolenfenster und in `scan2cad.log`.
Andere Systeme: `python -m venv .venv`, dann `pip install -r requirements.txt`, dann `python app/main.py`.

## Schnellstart

1. `beispiel/testteil.stl` öffnen (oder den eigenen Scan).
2. In der Werkzeugleiste **✨ Auf gut Glück** klicken. Der Autopilot fährt alle Schritte ab (Reparieren → Ausrichten → Erkennen → Bohrungen → Symmetrie → STEP-Vorschau) und hält nach jedem an. Auf den Seiten links korrigieren, dann **▶ Weiter**.
3. Bohrungen auf Seite **5 · Bohrungen** prüfen (Tiefe, durchgehend / Grundloch, Gewinde), Symmetrien und Bedingungen auf Seite **6**.
4. Auf Seite **7 · Export** die **STEP-Vorschau (P)** öffnen, den Abgleich im Protokoll lesen, dann **STEP exportieren…**.

Ein zweites Beispiel mit offenem Boden, wie vom Drehteller-Scanner, liegt in `beispiel/flansch_offen.stl`. Die Sollmaße stehen in der ausführlichen Dokumentation.

## Gut zu wissen

- **Prismatische Teile und Drehteile** (gefräst, gedruckt, gespritzt) gehen am besten. Bei den Testteilen sind typischerweise alle Features exakt, und das Volumen weicht 0,05–0,12 % vom CAD-Original ab.
- **Stark verrundete Freiform-Teile** (Spielzeug, Gehäuse) werden ebenfalls geschlossen, Teile davon aber als Freiform-Flächen oder als Scan-Dreiecke. Der Abgleich zeigt genau, wo.
- **Was der Scanner nicht sieht** (Bohrungsgrund, wahre Lage eines offenen Bodens), wird angenommen und markiert. Die Entscheidung liegt bei dir.
- **Noch nicht:** automatische Erkennung von Extrusionen (Profil × Höhe) und von Fasen.
- **Kritische Maße immer selbst prüfen**, bevor nach einem Scan gefertigt wird.

## Dokumentation

Die ausführliche Dokumentation steht in [Scan2CAD_V1.0.1/Scan2CAD/README.md](Scan2CAD_V1.0.1/Scan2CAD/README.md): alle Seiten und Tasten, die Schritte von „Auf gut Glück“, das Bohrungswerkzeug, wie der geschlossene STEP entsteht, der Abgleich Features ↔ STEP und die Tests.

## Lizenz und Zitieren

Copyright © 2026 **Amon P. Lanz**. Lizenz: **[Apache License 2.0](LICENSE)**, siehe [NOTICE](Scan2CAD_V1.0.1/Scan2CAD/NOTICE).

Ein großer Teil des Codes wurde mit **Claude Opus 5.5** (Anthropic) geschrieben. Idee, Scans, Testteile, Kritik und jede Entscheidung, was gut genug ist, kamen von Amon P. Lanz.

Zitieren ist freiwillig und willkommen:

> Amon P. Lanz (2026). *Scan2CAD: Guided Reverse Engineering of 3D Scans to STEP* (Version 1.0.1). Computer software.
> https://github.com/AmLanz/Scan2CAD

Im Programm kopiert **Über / Zitieren** (rechte obere Ecke) das Zitat oder einen BibTeX-Eintrag. [CITATION.cff](Scan2CAD_V1.0.1/Scan2CAD/CITATION.cff) und [CITATION.bib](Scan2CAD_V1.0.1/Scan2CAD/CITATION.bib) liegen bei.
