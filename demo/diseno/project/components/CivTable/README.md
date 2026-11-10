Tabla de civilizaciones de un episodio: una fila por civilización con su color, su estado, sus recursos y la acción del turno actual. Usa `ob-table`; el nombre va en `ob-civ` con un disco del color `civ-N` fijado en `--c`.

**El consumidor aporta** por fila: el índice (1 a 8), si está viva, los recursos con dos decimales y el nombre de la acción («crecer», «compartir», «mentir», «refugiarse»). Una civilización eliminada lleva `data-dead="true"` en la fila y en `ob-civ`.

**Estados.** Fila viva con texto `ink`; fila eliminada con texto `muted`, disco `dead` y la palabra «muerto» en la columna de estado. Las cifras van alineadas a la derecha en mono.

**Accesibilidad.** Es una `<table>` con `<th scope="col">`. El estado se dice con palabra («vivo»/«muerto»), nunca solo con el color del disco. Anuncia los cambios de turno con una región `aria-live="off"` para no saturar al lector.

| Haz | No hagas |
|---|---|
| Alinea las cifras a la derecha en `data`. | Mezclar cifras con texto en la misma celda. |
| Escribe «mentir» cuando la acción sea una mentira. | Mostrar el identificador numérico de la acción. |
