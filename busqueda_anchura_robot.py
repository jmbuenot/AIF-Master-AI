#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Búsqueda en anchura (BFS) para un robot perforador con orientación.
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
  Blind search (BFS): (d, g(n), op, S)
  donde S = (fila, columna, orientación)
- Totales: número de nodos explorados (expanded) y tamaño final de la frontera.

Nota: BFS expande por profundidad (nº de pasos), no por coste acumulado. Por tanto, no garantiza
la optimalidad en coste cuando los costes de acciones difieren. Aquí acumulamos g(n) para informar
y cumplir el formato, pero el orden de expansión es por capas.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import sys
import re

# Mapeo de orientaciones (0..7) a desplazamientos (dr, dc)
# 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW
DIRS: List[Tuple[int, int]] = [
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),   # E
    (1, 1),   # SE
    (1, 0),   # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1), # NW
]
DIR_NAMES = {
    0: "N",
    1: "NE",
    2: "E",
    3: "SE",
    4: "S",
    5: "SW",
    6: "W",
    7: "NW",
}

@dataclass(frozen=True)
class State:
    r: int
    c: int
    o: int  # 0..7

@dataclass
class Node:
    state: State
    parent: Optional["Node"]
    op: Optional[str]  # Operador que generó este nodo desde su padre
    depth: int         # d
    g: int             # coste acumulado g(n)

    def to_tuple(self) -> Tuple[int, int, Optional[str], Tuple[int, int, int]]:
        return (self.depth, self.g, self.op, (self.state.r, self.state.c, self.state.o))

# -------------------------------------------------------------
# Parsing de fichero de entrada
# -------------------------------------------------------------

def parse_ints(line: str) -> List[int]:
    # Acepta separadores espacio o coma
    parts = re.split(r"[\s,]+", line.strip())
    return [int(x) for x in parts if x != ""]

def read_problem(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip() != ""]
    if len(raw) < 4:
        raise ValueError("Fichero demasiado corto: faltan líneas para dimensiones, mapa e in/out")

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
    return grid, initial, goal

# -------------------------------------------------------------
# Sucesores
# -------------------------------------------------------------

def in_bounds(grid: List[List[int]], r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def successors(grid: List[List[int]], node: Node) -> List[Tuple[str, State, int]]:
    """Genera sucesores como (op, nuevo_estado, coste_op)."""
    r, c, o = node.state.r, node.state.c, node.state.o
    out: List[Tuple[str, State, int]] = []

    # Girar CW
    o_cw = (o + 1) % 8
    out.append(("TURN_CW", State(r, c, o_cw), 1))

    # Girar CCW
    o_ccw = (o - 1) % 8
    out.append(("TURN_CCW", State(r, c, o_ccw), 1))

    # Avanzar si es posible
    dr, dc = DIRS[o]
    nr, nc = r + dr, c + dc
    if in_bounds(grid, nr, nc):
        cost_move = grid[nr][nc]
        out.append(("MOVE", State(nr, nc, o), cost_move))
    # Si salir de la matriz, simplemente no se genera MOVE

    return out

# -------------------------------------------------------------
# Prueba de objetivo
# -------------------------------------------------------------

def is_goal(state: State, goal: State) -> bool:
    if state.r != goal.r or state.c != goal.c:
        return False
    if goal.o == 8:
        return True  # orientación indiferente
    return state.o == goal.o

# -------------------------------------------------------------
# BFS (búsqueda en anchura)
# -------------------------------------------------------------

def bfs(grid: List[List[int]], start: State, goal: State):
    start_node = Node(state=start, parent=None, op=None, depth=0, g=0)

    if is_goal(start, goal):
        return [start_node], [], 1  # camino trivial, frontera vacía, 1 explorado (o 0)

    frontier: deque[Node] = deque([start_node])
    explored: set[Tuple[int, int, int]] = set()

    # Para reconstruir camino cuando encontremos el objetivo
    goal_node: Optional[Node] = None

    while frontier:
        current = frontier.popleft()
        explored.add((current.state.r, current.state.c, current.state.o))

        for op, s_next, cost in successors(grid, current):
            if (s_next.r, s_next.c, s_next.o) in explored:
                continue

            child = Node(
                state=s_next,
                parent=current,
                op=op,
                depth=current.depth + 1,
                g=current.g + cost,
            )

            # Para evitar duplicados fuertes en la frontera, podemos llevar un set
            # con estados presentes en frontera. Alternativamente, comprobar aquí antes de añadir.
            already_in_frontier = any(
                (n.state.r, n.state.c, n.state.o) == (s_next.r, s_next.c, s_next.o)
                for n in frontier
            )
            if already_in_frontier:
                continue

            if is_goal(s_next, goal):
                goal_node = child
                # AUNQUE BFS podría seguir para recopilar estadísticas, 
                # normalmente devolvemos en cuanto encontramos la primera meta
                return reconstruct_path(goal_node), list(frontier), len(explored)

            frontier.append(child)

    # Sin solución: devolver el "mejor esfuerzo" (ninguno) y cifras
    return [], list(frontier), len(explored)

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


def print_trace_bfs(path: List[Node], frontier: List[Node], explored_count: int) -> None:
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

# -------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Uso: python busqueda_anchura_robot.py <ruta_al_fichero.txt>")
        return 1

    path = argv[1]
    grid, start, goal = read_problem(path)

    path_nodes, frontier, explored_count = bfs(grid, start, goal)

    # Imprimir traza
    print_trace_bfs(path_nodes, frontier, explored_count)

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

if __name__ == "__main__":
    sys.exit(main(sys.argv))
