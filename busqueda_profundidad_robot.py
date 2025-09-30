#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Búsqueda en profundidad (DFS) para un robot perforador con orientación.
Lee un fichero .txt con el formato descrito en el enunciado del laboratorio y en el mensaje del usuario:

Formato de entrada (texto plano):
1) Primera línea: "filas columnas" (separadas por espacio o coma). Ej.: 6 8
2) Siguientes 'filas' líneas: la matriz del mapa con dígitos 1..9 (separados por espacios).
3) Línea penúltima: estado inicial como "r0 c0 o0" (o con comas). La orientación inicial se asume 0 (Norte),
   pero se acepta la que aparezca en el fichero por comodidad.
4) Última línea: estado objetivo como "rt ct 8". La orientación objetivo puede ser 0..7, o 8 para "no importa".

Operadores disponibles en cada estado:
- TURN_CW (gira +45°, coste = 1)
- TURN_CCW (gira -45°, coste = 1)
- MOVE (avanza una casilla en la orientación actual, coste = dureza de la casilla destino)

Salida (traza mínima):
- Lista de nodos y operadores desde el inicial hasta el objetivo, con el formato exigido:
  Blind search (DFS): (d, g(n), op, S)
  donde S = (fila, columna, orientación)
- Totales: número de nodos explorados (expanded) y tamaño final de la frontera.

Nota: DFS prioriza seguir un camino hasta el fondo antes de retroceder, por lo que no garantiza
ni la optimalidad en pasos ni en coste. Aquí acumulamos g(n) para informar y cumplir el formato,
pero el orden de expansión responde a la disciplina LIFO de la pila.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import sys
import re

# Vectores de orientación en notación cartesiana (dx, dy) con Norte = (0, 1)
# 0 N, 1 NW, 2 W, 3 SW, 4 S, 5 SE, 6 E, 7 NE
ORIENTATION_VECTORS: List[Tuple[int, int]] = [
    (0, 1),    # N
    (-1, 1),   # NW
    (-1, 0),   # W
    (-1, -1),  # SW
    (0, -1),   # S
    (1, -1),   # SE
    (1, 0),    # E
    (1, 1),    # NE
]
# Explicación: almacena los desplazamientos (dx, dy) siguiendo el convenio pedido: Norte = (0, 1) y el resto en sentido antihorario.
DIR_NAMES = {
    0: "N",
    1: "NW",
    2: "W",
    3: "SW",
    4: "S",
    5: "SE",
    6: "E",
    7: "NE",
}
# Explicación: adapta las etiquetas human-readable al nuevo orden de orientaciones especificado por el enunciado.

@dataclass(frozen=True)
class State:
    """Estado elemental del robot en el mapa.

    Parámetros
    ----------
    r : int
        Índice de fila (0 = primera fila de la matriz).
    c : int
        Índice de columna (0 = primera columna de la matriz).
    o : int
        Orientación discreta del robot (0..7 siguiendo `ORIENTATION_VECTORS`).
    """

    r: int
    c: int
    o: int  # 0..7
# Explicación: encapsula la posición (fila y columna) y la orientación del robot en un estado inmutable; al ser `dataclass`
# con `frozen=True`, Python genera automáticamente un `__init__(r, c, o)` y lo hace apto para usarse como clave hash.

@dataclass
class Node:
    state: State
    parent: Optional["Node"]
    op: Optional[str]  # Operador que generó este nodo desde su padre
    depth: int         # d
    g: int             # coste acumulado g(n)

    def to_tuple(self) -> Tuple[int, int, Optional[str], Tuple[int, int, int]]:
        return (self.depth, self.g, self.op, (self.state.r, self.state.c, self.state.o))
# Explicación: describe un nodo del árbol de búsqueda con sus métricas y ofrece `to_tuple` para formatear la salida.

# -------------------------------------------------------------
# Parsing de fichero de entrada
# -------------------------------------------------------------

def parse_ints(line: str) -> List[int]:
    # Acepta separadores espacio o coma
    parts = re.split(r"[\s,]+", line.strip())
    return [int(x) for x in parts if x != ""]
# Explicación: transforma una línea en enteros admitiendo espacios o comas como separadores.

def read_problem(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip() != ""]
    if len(raw) < 4:
        raise ValueError("Fichero demasiado corto: faltan líneas para dimensiones, mapa e in/out")

# Explicación línea por línea de `read_problem`:
# 1. `def read_problem(path: str):` declara la función que recibirá la ruta al fichero de entrada.
# 2. `with open(path, "r", encoding="utf-8") as f:` abre el fichero en modo lectura utilizando UTF-8.
# 3. `raw = [ln.strip() for ln in f if ln.strip() != ""]` crea una lista sin líneas vacías, eliminando espacios.
# 4. `if len(raw) < 4:` verifica que, al menos, existan dimensiones, mapa e in/out.
# 5. `raise ValueError(...)` lanza una excepción si el fichero no contiene los bloques mínimos requeridos.

    # 1) Dimensiones
    rows, cols = parse_ints(raw[0])

    # 2) Matriz
    grid: List[List[int]] = []
    for i in range(1, 1 + rows):
        row_vals = parse_ints(raw[i])
        if len(row_vals) != cols:
            raise ValueError(f"Línea {i+1}: se esperaban {cols} valores, llegaron {len(row_vals)}")
        if any(v < 1 or v > 9 for v in row_vals):
            raise ValueError(f"Línea {i+1}: todos los valores deben ser dígitos 1..9")
        grid.append(row_vals)

    # 3) Estados inicial y objetivo
    init_line = parse_ints(raw[1 + rows])
    goal_line = parse_ints(raw[2 + rows])
    if len(init_line) != 3 or len(goal_line) != 3:
        raise ValueError("Las líneas de estado deben ser triples: r c o")

    r0, c0, o0 = init_line
    rt, ct, ot = goal_line
    # Explicación: `parse_ints` devuelve listas de enteros y aquí aplicamos el patrón de
    # desempaquetado de secuencias de Python (sequence unpacking) para asignar cada
    # componente de las triples `init_line` y `goal_line` a variables con nombres
    # descriptivos. No interviene ninguna clase adicional: es una característica del
    # lenguaje que funciona con cualquier iterable con la longitud adecuada.

    # Validaciones básicas
    for (r, c, name) in [(r0, c0, "inicio"), (rt, ct, "objetivo")]:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"Coordenadas de {name} fuera de rango: {(r, c)}")
    if not (0 <= o0 <= 7):
        raise ValueError("La orientación inicial debe estar en 0..7")
    if not (0 <= ot <= 8):
        raise ValueError("La orientación del objetivo debe estar en 0..8 (8 = no importa)")

    initial = State(r0, c0, o0)
    goal = State(rt, ct, ot)  # si ot==8, no importa la orientación
    # Explicación: la clase `State` es un `@dataclass`, por lo que Python crea de forma implícita un constructor
    # `State(r, c, o)`. Tras desempaquetar las coordenadas, basta con pasar cada componente en orden para obtener las
    # instancias `initial` y `goal` sin escribir un `__init__` manual.
    return grid, initial, goal
# Explicación: carga el fichero, valida dimensiones y coordenadas, y genera los estados inicial y objetivo junto al mapa.

# -------------------------------------------------------------
# Visualización del problema
# -------------------------------------------------------------

def print_problem_overview(grid: List[List[int]], start: State, goal: State) -> None:
    print("Mapa (matriz de durezas):")
    for idx, row in enumerate(grid):
        values = " ".join(str(v) for v in row)
        print(f"Fila {idx:02d}: {values}")

    start_dir = DIR_NAMES.get(start.o, f"{start.o}")
    if goal.o == 8:
        goal_dir = "indiferente"
    else:
        goal_dir = DIR_NAMES.get(goal.o, f"{goal.o}")

    print(
        f"Estado inicial -> fila: {start.r}, columna: {start.c}, orientación: {start_dir}"
    )
    print(
        f"Estado objetivo -> fila: {goal.r}, columna: {goal.c}, orientación: {goal_dir}"
    )
# Explicación: muestra la matriz numerada por filas y resume las coordenadas iniciales y finales usando etiquetas de orientación legibles.

# -------------------------------------------------------------
# Orquestación de lectura y búsqueda
# -------------------------------------------------------------

def solve_from_file(
    path: str,
    *,
    show_overview: bool = True,
) -> Tuple[List[List[int]], State, State, List[Node], List[Node], int]:
    """Lee un problema desde disco, opcionalmente lo muestra y ejecuta la DFS."""

    grid, start, goal = read_problem(path)

    if show_overview:
        print_problem_overview(grid, start, goal)
        print()  # Línea en blanco para separar del resto de la salida

    path_nodes, frontier, explored_count = dfs(grid, start, goal)
    return grid, start, goal, path_nodes, frontier, explored_count
# Explicación: función de alto nivel que encapsula la lectura del fichero, el despliegue del mapa y la invocación de la DFS devolviendo todos los resultados necesarios.

# -------------------------------------------------------------
# Sucesores
# -------------------------------------------------------------

def in_bounds(grid: List[List[int]], r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])
# Explicación: verifica si unas coordenadas caen dentro de los límites de la cuadrícula.


def successors(grid: List[List[int]], node: Node) -> List[Tuple[str, State, int]]:
    """Genera sucesores como (op, nuevo_estado, coste_op)."""
    r, c, o = node.state.r, node.state.c, node.state.o
    out: List[Tuple[str, State, int]] = []

    # Avanzar si es posible
    dx, dy = ORIENTATION_VECTORS[o]
    dr, dc = -dy, dx  # convertir de notación cartesiana (x, y) a índices de matriz (fila, columna)
    nr, nc = r + dr, c + dc
    if in_bounds(grid, nr, nc):
        cost_move = grid[nr][nc]
        out.append(("MOVE", State(nr, nc, o), cost_move))
    # Si salir de la matriz, simplemente no se genera MOVE

    # Girar CW (el índice disminuye porque el orden de orientaciones avanza en sentido antihorario)
    o_cw = (o - 1) % 8
    out.append(("TURN_CW", State(r, c, o_cw), 1))

    # Girar CCW
    o_ccw = (o + 1) % 8
    out.append(("TURN_CCW", State(r, c, o_ccw), 1))


    return out
# Explicación: lista giros y movimientos factibles; al avanzar traduce el vector cartesiano (dx, dy) de la orientación al desplazamiento (fila, columna) y descarta pasos que salgan del mapa.

# -------------------------------------------------------------
# Prueba de objetivo
# -------------------------------------------------------------

def is_goal(state: State, goal: State) -> bool:
    if state.r != goal.r or state.c != goal.c:
        return False
    if goal.o == 8:
        return True  # orientación indiferente
    return state.o == goal.o
# Explicación: comprueba si la posición coincide con la meta y valida la orientación salvo que sea indiferente.

# -------------------------------------------------------------
# DFS (búsqueda en profundidad)
# -------------------------------------------------------------

def dfs(grid: List[List[int]], start: State, goal: State):
    """Ejecuta una búsqueda en profundidad iterativa sobre el espacio de estados del robot.

    Se devuelve una tupla con:
        * La lista de nodos desde el origen hasta la meta (inclusive) si hay solución.
        * El contenido restante de la frontera al terminar.
        * El número de estados explorados.
    """

    start_node = Node(state=start, parent=None, op=None, depth=0, g=0)

    if is_goal(start, goal):
        return [start_node], [], 1  # camino trivial, frontera vacía, 1 explorado (o 0)

    frontier: List[Node] = [start_node]
    frontier_states: set[Tuple[int, int, int]] = {
        (start.r, start.c, start.o)
    }
    # Explicación: `frontier` es ahora una pila (lista LIFO) que profundiza antes de retroceder; el
    # conjunto auxiliar `frontier_states` registra los estados actualmente apilados para detectar
    # duplicados sin recorrer la estructura completa.
    explored: set[Tuple[int, int, int]] = set()

    while frontier:
        current = frontier.pop()
        frontier_states.discard((current.state.r, current.state.c, current.state.o))
        # Explicación: `discard` elimina el estado de la pila si está presente y evita excepciones si ya no figura en el conjunto.
        explored.add((current.state.r, current.state.c, current.state.o))

        for op, s_next, cost in successors(grid, current):
            state_key = (s_next.r, s_next.c, s_next.o)
            if state_key in explored or state_key in frontier_states:
                continue

            child = Node(
                state=s_next,
                parent=current,
                op=op,
                depth=current.depth + 1,
                g=current.g + cost,
            )

            if is_goal(s_next, goal):
                return reconstruct_path(child), list(frontier), len(explored)

            frontier.append(child)
            frontier_states.add(state_key)

    # Sin solución: devolver el "mejor esfuerzo" (ninguno) y cifras
    return [], list(frontier), len(explored)
# Explicación: ejecuta la DFS gestionando pila y explorados y devuelve el camino hallado junto a estadísticas.

# -------------------------------------------------------------
# Utilidades de camino y trazas
# -------------------------------------------------------------

def reconstruct_path(node: Node) -> List[Node]:
    path: List[Node] = []
    cur: Optional[Node] = node
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    path.reverse()
    return path
# Explicación: recompone la secuencia de nodos siguiendo los punteros al padre desde la meta hasta el inicio.


def print_trace_dfs(path: List[Node], frontier: List[Node], explored_count: int) -> None:
    if not path:
        print("No se ha encontrado solución.")
        print(f"Total explorados: {explored_count}")
        print(f"Total en frontera: {len(frontier)}")
        return

    # Formato tipo Sección 4.2 del enunciado
    for i, node in enumerate(path):
        if i == 0:
            print(f"Node {i} (starting node)")
        else:
            print(f"Node {i}")
        d, g, op, (r, c, o) = node.to_tuple()
        # Blind search: (d, g(n), op, S)
        op_disp = op if op is not None else "START"
        print(f"(d={d}, g(n)={g}, op={op_disp}, S=({r}, {c}, {o}:{DIR_NAMES[o]}))")
        if i < len(path) - 1:
            next_op = path[i + 1].op or "?"
            print(f"Operator {i}: {next_op}")

    print()
    print(f"Total number of items in explored list: {explored_count}")
    print(f"Total number of items in frontier: {len(frontier)}")
# Explicación: muestra la traza paso a paso o informa de la ausencia de solución junto a métricas finales.

# -------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Uso: python busqueda_profundidad_robot.py <ruta_al_fichero.txt>")
        return 1

    path = argv[1]
    (
        grid,
        start,
        goal,
        path_nodes,
        frontier,
        explored_count,
    ) = solve_from_file(path, show_overview=True)
# Explicación: delega la lectura del fichero, el despliegue del mapa y la ejecución de la DFS en `solve_from_file` para reutilizar la lógica y asegurar el uso de `print_problem_overview`.

    # Imprimir traza
    print_trace_dfs(path_nodes, frontier, explored_count)

    # Devolver (por si se importa como módulo): lista de pasos (op), y nodos en formato solicitado
    if path_nodes:
        actions = [n.op for n in path_nodes if n.op is not None]
        nodes_fmt = [n.to_tuple() for n in path_nodes]
    else:
        actions = []
        nodes_fmt = []

    # Mostrar resumen compacto adicional
    print("\nResumen compacto:")
    print("Acciones:", actions)
    print("Nodos (d, g, op, (r,c,o)):")
    for t in nodes_fmt:
        print(t)

    return 0
# Explicación: orquesta la interacción por consola leyendo datos, ejecutando DFS y presentando resultados detallados.

if __name__ == "__main__":
    sys.exit(main(sys.argv))
# Explicación: permite ejecutar el módulo como script invocando `main` con los argumentos de la línea de comandos.