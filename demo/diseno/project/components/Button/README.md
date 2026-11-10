Botón de acción. Hay una acción principal por panel (`ob-btn--primary`), acciones secundarias (`ob-btn`) y terciarias sin relleno (`ob-btn--ghost`).

**El consumidor aporta** el texto del botón, en infinitivo («Simular», «Reproducir»), y el manejador de clic. Para una acción larga marca `aria-busy="true"` y cambia el texto a «Simulando…».

| Variante | Cuándo |
|---|---|
| `ob-btn--primary` | La acción que avanza la tarea: Simular. Una por panel. |
| `ob-btn` | Acciones de apoyo: Pausa, paso anterior, paso siguiente. |
| `ob-btn--ghost` | Acciones de menor peso dentro de un grupo denso. |

**Estados.** En reposo; `:hover` (borde `ink`, y resplandor `shadow-glow` en la principal); `:focus-visible` con el anillo `focus`; `disabled` y `aria-busy` con opacidad 0.55 y cursor `not-allowed`.

**Accesibilidad.** Es un `<button>` nativo; Enter y Espacio lo activan. Alto mínimo de 40 px. Un botón de solo glifo (`◀`) necesita `aria-label`.

| Haz | No hagas |
|---|---|
| Usa el texto de la acción: «Simular». | Usar «Aceptar» o «OK». |
| Un solo botón principal por panel. | Dos botones `sun` juntos. |
