package causalrag.causalgraph.graph

/**
 * Lightweight directed graph with weighted edges.
 */
class DirectedGraph {
    private val outEdges: MutableMap<String, MutableMap<String, Double>> = mutableMapOf()
    private val inEdges: MutableMap<String, MutableSet<String>> = mutableMapOf()

    /**
     * Adds or updates a directed edge.
     *
     * @param from Source node identifier.
     * @param to Target node identifier.
     * @param weight Edge weight.
     */
    fun addEdge(
        from: String,
        to: String,
        weight: Double,
    ) {
        val targets = outEdges.getOrPut(from) { mutableMapOf() }
        targets[to] = weight
        inEdges.getOrPut(to) { mutableSetOf() }.add(from)
        outEdges.getOrPut(to) { mutableMapOf() }
        inEdges.getOrPut(from) { mutableSetOf() }
    }

    /**
     * Returns the direct successors of a node.
     *
     * @param node Node identifier.
     * @return Successor node identifiers.
     */
    fun successors(node: String): Set<String> = outEdges[node]?.keys ?: emptySet()

    /**
     * Returns the direct predecessors of a node.
     *
     * @param node Node identifier.
     * @return Predecessor node identifiers.
     */
    fun predecessors(node: String): Set<String> = inEdges[node] ?: emptySet()

    /**
     * Returns all node identifiers in the graph.
     *
     * @return Node identifier set.
     */
    fun nodes(): Set<String> = outEdges.keys

    /**
     * Returns the number of nodes in the graph.
     *
     * @return Node count.
     */
    fun numberOfNodes(): Int = outEdges.size

    /**
     * Returns the number of edges in the graph.
     *
     * @return Edge count.
     */
    fun numberOfEdges(): Int = outEdges.values.sumOf { it.size }

    /**
     * Returns the in-degree of a node.
     *
     * @param node Node identifier.
     * @return Number of incoming edges.
     */
    fun inDegree(node: String): Int = inEdges[node]?.size ?: 0

    /**
     * Returns the out-degree of a node.
     *
     * @param node Node identifier.
     * @return Number of outgoing edges.
     */
    fun outDegree(node: String): Int = outEdges[node]?.size ?: 0

    /**
     * Returns all graph edges.
     *
     * @return Edge list.
     */
    fun edges(): List<Edge> {
        val result = mutableListOf<Edge>()
        for ((from, targets) in outEdges) {
            for ((to, weight) in targets) {
                result.add(Edge(from, to, weight))
            }
        }
        return result
    }

    /**
     * Builds a subgraph containing only the supplied nodes.
     *
     * @param nodeSet Nodes to retain.
     * @return Induced subgraph.
     */
    fun subgraph(nodeSet: Set<String>): DirectedGraph {
        val sub = DirectedGraph()
        val retained = nodeSet.intersect(nodes())
        for (node in retained) {
            sub.outEdges.getOrPut(node) { mutableMapOf() }
            sub.inEdges.getOrPut(node) { mutableSetOf() }
        }
        for (edge in edges()) {
            if (edge.from in retained && edge.to in retained) {
                sub.addEdge(edge.from, edge.to, edge.weight)
            }
        }
        return sub
    }

    /**
     * Returns the weight of an edge if it exists.
     *
     * @param from Source node identifier.
     * @param to Target node identifier.
     * @return Edge weight or `null` when absent.
     */
    fun edgeWeight(
        from: String,
        to: String,
    ): Double? = outEdges[from]?.get(to)

    /**
     * Returns the maximum edge weight in the graph.
     *
     * @return Maximum edge weight, or `null` for an empty graph.
     */
    fun maxEdgeWeight(): Double? = edges().maxOfOrNull { it.weight }

    /**
     * Removes all nodes and edges from the graph.
     */
    fun clear() {
        outEdges.clear()
        inEdges.clear()
    }

    /**
     * Creates a structural copy of the graph.
     *
     * @return Independent graph copy with the same nodes and edges.
     */
    fun copy(): DirectedGraph {
        val clone = DirectedGraph()
        for (node in nodes()) {
            clone.outEdges.getOrPut(node) { mutableMapOf() }
            clone.inEdges.getOrPut(node) { mutableSetOf() }
        }
        for (edge in edges()) {
            clone.addEdge(edge.from, edge.to, edge.weight)
        }
        return clone
    }

    /**
     * Reports whether the graph is acyclic.
     *
     * @return `true` when the graph is a DAG.
     */
    fun isDag(): Boolean = !hasCycle()

    /**
     * Reports whether the graph contains a directed cycle.
     *
     * @return `true` when any directed cycle exists.
     */
    fun hasCycle(): Boolean {
        val visited = mutableSetOf<String>()
        val stack = mutableSetOf<String>()

        fun dfs(node: String): Boolean {
            if (node in stack) return true
            if (node in visited) return false
            visited.add(node)
            stack.add(node)
            for (next in successors(node)) {
                if (dfs(next)) return true
            }
            stack.remove(node)
            return false
        }

        for (node in nodes()) {
            if (dfs(node)) return true
        }
        return false
    }

    /**
     * Finds weakly connected components.
     *
     * @return Components represented as node sets.
     */
    fun weaklyConnectedComponents(): List<Set<String>> {
        val remaining = nodes().toMutableSet()
        val components = mutableListOf<Set<String>>()
        while (remaining.isNotEmpty()) {
            val start = remaining.first()
            val queue = ArrayDeque<String>()
            val component = mutableSetOf<String>()
            queue.add(start)
            remaining.remove(start)
            component.add(start)
            while (queue.isNotEmpty()) {
                val current = queue.removeFirst()
                val neighbors = successors(current) + predecessors(current)
                for (n in neighbors) {
                    if (n in remaining) {
                        remaining.remove(n)
                        component.add(n)
                        queue.add(n)
                    }
                }
            }
            components.add(component)
        }
        return components
    }

    /**
     * Finds simple directed paths between two nodes.
     *
     * @param start Start node.
     * @param end End node.
     * @param maxDepth Maximum path depth to explore.
     * @param limit Maximum number of paths to return.
     * @return Paths from [start] to [end].
     */
    fun findPaths(
        start: String,
        end: String,
        maxDepth: Int,
        limit: Int = Int.MAX_VALUE,
    ): List<List<String>> {
        val results = mutableListOf<List<String>>()

        fun dfs(
            current: String,
            target: String,
            depth: Int,
            path: MutableList<String>,
            visited: MutableSet<String>,
        ) {
            if (current == target) {
                results.add(path.toList())
                return
            }
            if (depth >= maxDepth || results.size >= limit) return
            for (next in successors(current)) {
                if (next in visited) continue
                visited.add(next)
                path.add(next)
                dfs(next, target, depth + 1, path, visited)
                path.removeAt(path.size - 1)
                visited.remove(next)
                if (results.size >= limit) return
            }
        }
        dfs(start, end, 0, mutableListOf(start), mutableSetOf(start))
        return results
    }
}

/**
 * Weighted directed edge.
 *
 * @property from Source node identifier.
 * @property to Target node identifier.
 * @property weight Edge weight.
 */
data class Edge(
    val from: String,
    val to: String,
    val weight: Double,
)
