import heapq

def a_star(root, goal, graph, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, root))

    g_cost = {root: 0}
    parent = {root: None}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for child, cost in graph[current]:
            new_g = g_cost[current] + cost

            if child not in g_cost or new_g < g_cost[child]:
                g_cost[child] = new_g
                f_cost = new_g + heuristic[child]
                heapq.heappush(open_list, (f_cost, child))
                parent[child] = current

    return None


# Tree structure (root finding)
graph = {
    'A': [('B', 1), ('C', 2)],
    'B': [('D', 3)],
    'C': [('G', 2)],
    'D': [],
    'G': []
}

# Heuristic values
heuristic = {
    'A': 5,
    'B': 4,
    'C': 2,
    'D': 6,
    'G': 0
}

root = 'A'
goal = 'G'

path = a_star(root, goal, graph, heuristic)
print("Path from root to goal:", path)