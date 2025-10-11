# ProjectTimeline

Example CSV (`example_tasks.csv`) now includes an optional `marker` column for milestones. Values correspond to [Matplotlib marker symbols](https://matplotlib.org/stable/api/markers_api.html) such as:

- `v` — triangle down (default)
- `o` — circle
- `s` — square
- `^` — triangle up

Use these markers to customize milestone icons in the Gantt chart.

The sidebar offers extensive customization for dependency arrows:

- choose between curved or orthogonal connectors
- adjust curvature, color, transparency, line width and head size
- select arrow head style (e.g., `-|>`, `->`, `-[`)

Milestone titles displayed on vertical lines can be placed above or below the chart to keep long labels from overlapping tasks.

Users may also align task titles either centered or starting at the beginning of their bars to keep starts lined up.

Date format fields in the sidebar now offer a select box of common display options with an extra *Personnalisé…* entry for custom `strftime` strings.

## Excel imports & congés

Exporter les données depuis l'éditeur Streamlit génère désormais un fichier `donnees.xlsx` contenant plusieurs feuilles :

- `planning` pour les activités et jalons.
- `inputs` pour les annotations optionnelles.
- `leave_zones` pour les périodes de congés (colonnes `start`, `end`, `label`).
- `jours_feries` pour les jours fériés (colonne `date`).

Pour réimporter automatiquement vos congés, remplissez simplement la feuille `leave_zones` avec vos dates (`start`/`end`) et un libellé facultatif. Lors du chargement d'un fichier Excel, ces informations sont reprises et pré-remplies dans la section « Zones hachurées » de l'application, évitant ainsi de devoir les ressaisir à chaque fois.
