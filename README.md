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
