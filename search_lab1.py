import sys
import heapq
from collections import deque

# Orientation codes and their corresponding unit movement vectors (dx, dy)
orientations = {
    0: (-1, 0),   # North
    1: (-1, 1),   # Northeast
    2: (0, 1),    # East
    3: (1, 1),    # Southeast
    4: (1, 0),    # South
    5: (1, -1),   # Southwest
    6: (0, -1),   # West
    7: (-1, -1)   # Northwest
}

def get_neighbors(state, terrain):
    """
    Generate successor states for a given state (x, y, o) on the terrain.
    Returns (next_state, operator, step_cost) for each valid action.
    """
    x, y, o = state
    max_row = len(terrain) - 1
    max_col = len(terrain[0]) - 1
    # Move forward in current orientation (if within bounds and not impassable)
    dx, dy = orientations[o]
    nx, ny = x + dx, y + dy
    if 0 <= nx <= max_row and 0 <= ny <= max_col:
        if terrain[nx][ny] < 99:  # treat hardness 99+ as impassable
            yield ((nx, ny, o), 'MF', terrain[nx][ny])
    # Rotate 45º clockwise (in place)
    yield ((x, y, (o + 1) % 8), 'RCW', 1)
    # Rotate 45º counterclockwise (in place)
    yield ((x, y, (o - 1) % 8), 'RCCW', 1)

def heuristic(state, goal):
    """
    Heuristic for A*: Chebyshev distance (max of dx,dy) plus minimal rotations to face goal direction.
    This never overestimates the true cost.
    """
    x, y, o = state
    gx, gy, go = goal
    # Compute Chebyshev distance (min number of moves ignoring orientation)
    dx = abs(gx - x)
    dy = abs(gy - y)
    D = max(dx, dy)
    if D == 0:
        return 0  # already at goal position
    # Determine ideal orientation towards goal position (one of 8 directions)
    if x < gx and y < gy:
        desired_o = 3  # SE
    elif x < gx and y > gy:
        desired_o = 5  # SW
    elif x > gx and y < gy:
        desired_o = 1  # NE
    elif x > gx and y > gy:
        desired_o = 7  # NW
    elif x < gx and y == gy:
        desired_o = 4  # S
    elif x > gx and y == gy:
        desired_o = 0  # N
    elif x == gx and y < gy:
        desired_o = 2  # E
    else:  # x == gx and y > gy
        desired_o = 6  # W
    # Minimal 45º rotations from current orientation to desired orientation
    diff = abs(desired_o - o) % 8
    if diff > 4:
        diff = 8 - diff
    return D + diff

def bfs_search(terrain, start, goal):
    """Breadth-First Search: returns (path, (explored_count, frontier_count))."""
    frontier = deque([start])
    explored = set([start])
    parent = {start: None}
    op_taken = {start: None}
    explored_count = 0
    last_state = None
    gx, gy, go = goal
    while frontier:
        current = frontier.popleft()
        explored_count += 1
        last_state = current
        x, y, o = current
        if x == gx and y == gy and (go == 8 or o == go):
            # Goal found
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1], (explored_count, len(frontier))
        # Expand neighbors
        for nbr, op, cost in get_neighbors(current, terrain):
            if nbr not in explored:
                explored.add(nbr)
                parent[nbr] = current
                op_taken[nbr] = op
                frontier.append(nbr)
    # No solution found; return path to last expanded node
    path = []
    node = last_state
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1], (explored_count, len(frontier))

def dfs_search(terrain, start, goal):
    """Depth-First Search: returns (path, (explored_count, frontier_count))."""
    frontier = [start]
    explored = set([start])
    parent = {start: None}
    op_taken = {start: None}
    explored_count = 0
    last_state = None
    gx, gy, go = goal
    while frontier:
        current = frontier.pop()
        explored_count += 1
        last_state = current
        x, y, o = current
        if x == gx and y == gy and (go == 8 or o == go):
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1], (explored_count, len(frontier))
        # Expand neighbors (push in reverse order for correct DFS traversal order)
        neighbors = list(get_neighbors(current, terrain))
        for nbr, op, cost in reversed(neighbors):
            if nbr not in explored:
                explored.add(nbr)
                parent[nbr] = current
                op_taken[nbr] = op
                frontier.append(nbr)
    path = []
    node = last_state
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1], (explored_count, len(frontier))

def astar_search(terrain, start, goal):
    """A* Search: returns (optimal_path, (explored_count, frontier_count))."""
    # Priority queue holds (f = g+h, g, state)
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), 0, start))
    g_cost = {start: 0}
    parent = {start: None}
    op_taken = {start: None}
    explored_count = 0
    closed = set()
    last_state = None
    gx, gy, go = goal
    while frontier:
        f, g, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if g > g_cost.get(current, float('inf')):
            continue  # outdated entry
        explored_count += 1
        last_state = current
        closed.add(current)
        x, y, o = current
        if x == gx and y == gy and (go == 8 or o == go):
            # Goal reached; reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1], (explored_count, sum(1 for (_, _, s) in frontier if s not in closed))
        # Expand neighbors
        for nbr, op, cost in get_neighbors(current, terrain):
            new_g = g_cost[current] + cost
            if nbr in closed and new_g < g_cost.get(nbr, float('inf')):
                closed.remove(nbr)  # allow reexploration
            if nbr not in closed and new_g < g_cost.get(nbr, float('inf')):
                g_cost[nbr] = new_g
                parent[nbr] = current
                op_taken[nbr] = op
                new_f = new_g + heuristic(nbr, goal)
                heapq.heappush(frontier, (new_f, new_g, nbr))
    # No solution; return path to last expanded node
    path = []
    node = last_state
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1], (explored_count, sum(1 for (_, _, s) in frontier if s not in closed))

def main():
    if len(sys.argv) < 3:
        print("Usage: python search_lab1.py <map_file> <method>")
        print("Methods: bfs | dfs | astar")
        return
    map_file = sys.argv[1]
    method = sys.argv[2].lower()
    # Read terrain map from file
    try:
        with open(map_file, 'r') as f:
            content = f.read().strip().split()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    it = iter(content)
    try:
        rows = int(next(it)); cols = int(next(it))
    except StopIteration:
        print("Invalid map file format.")
        return
    terrain = [[int(next(it)) for _ in range(cols)] for _ in range(rows)]
    try:
        sx, sy, so = int(next(it)), int(next(it)), int(next(it))
        gx, gy, go = int(next(it)), int(next(it)), int(next(it))
    except StopIteration:
        print("Map file missing start/goal info.")
        return
    start_state = (sx, sy, so)
    goal_state = (gx, gy, go)
    # Run chosen search algorithm
    if method == 'bfs':
        path, (exp, fr) = bfs_search(terrain, start_state, goal_state)
    elif method == 'dfs':
        path, (exp, fr) = dfs_search(terrain, start_state, goal_state)
    elif method in ('astar', 'a*'):
        path, (exp, fr) = astar_search(terrain, start_state, goal_state)
    else:
        print("Unknown method. Use bfs, dfs, or astar.")
        return
    # Output execution trace
    if not path or not (path[-1][0] == gx and path[-1][1] == gy):
        print("No solution found.")
        if path:
            print("Path to last explored node:")
        node_index = 0
        total_cost = 0
        for state in path:
            x, y, o = state
            op = '-' if node_index == 0 else None
            if node_index > 0:  # determine operator for this state
                px, py, po = path[node_index - 1]
                if x == px and y == py:
                    # Orientation changed (rotate)
                    op = 'RCW' if ((o - po) % 8 == 1 or (po - o) % 8 == 7) else 'RCCW'
                    step_cost = 1
                else:
                    op = 'MF'
                    step_cost = terrain[x][y]
                total_cost += step_cost
            depth = node_index
            g_val = total_cost
            if method in ('astar', 'a*'):
                h_val = heuristic(state, goal_state)
                print(f"Node {node_index}: ({depth}, {g_val}, {op}, {h_val}, ({x},{y},{o}))")
            else:
                print(f"Node {node_index}: ({depth}, {g_val}, {op}, ({x},{y},{o}))")
            node_index += 1
        print(f"Total number of items in explored list: {exp}")
        print(f"Total number of items in frontier: {fr}")
        return
    # If solution found:
    total_cost = 0
    for i, (x, y, o) in enumerate(path):
        if i == 0:
            op = '-'
            depth = 0
            g_val = 0
        else:
            px, py, po = path[i-1]
            if x == px and y == py:
                op = 'RCW' if ((o - po) % 8 == 1 or (po - o) % 8 == 7) else 'RCCW'
                step_cost = 1
            else:
                op = 'MF'
                step_cost = terrain[x][y]
            total_cost += step_cost
            depth = i
            g_val = total_cost
        if method in ('astar', 'a*'):
            h_val = heuristic((x, y, o), goal_state)
            print(f"Node {i}: ({depth}, {g_val}, {op}, {h_val}, ({x},{y},{o}))")
        else:
            print(f"Node {i}: ({depth}, {g_val}, {op}, ({x},{y},{o}))")
    print(f"Total number of items in explored list: {exp}")
    print(f"Total number of items in frontier: {fr}")

if __name__ == "__main__":
    main()