package causalrag.causalgraph.graph

import kotlin.math.max

class DirectedGraph {
    private val outEdges: MutableMap<String, MutableMap<String, Double>> = mutableMapOf()
    private val inEdges: MutableMap<String, MutableSet<String>> = mutableMapOf()

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

    fun successors(node: String): Set<String> = outEdges[node]?.keys ?: emptySet()

    fun predecessors(node: String): Set<String> = inEdges[node] ?: emptySet()

    fun nodes(): Set<String> = outEdges.keys

    fun numberOfNodes(): Int = outEdges.size

    fun numberOfEdges(): Int = outEdges.values.sumOf { it.size }

    fun inDegree(node: String): Int = inEdges[node]?.size ?: 0

    fun outDegree(node: String): Int = outEdges[node]?.size ?: 0

    fun edges(): List<Edge> {
        val result = mutableListOf<Edge>()
        for ((from, targets) in outEdges) {
            for ((to, weight) in targets) {
                result.add(Edge(from, to, weight))
            }
        }
        return result
    }

    fun subgraph(nodeSet: Set<String>): DirectedGraph {
        val sub = DirectedGraph()
        for (edge in edges()) {
            if (edge.from in nodeSet && edge.to in nodeSet) {
                sub.addEdge(edge.from, edge.to, edge.weight)
            }
        }
        return sub
    }

    fun edgeWeight(
        from: String,
        to: String,
    ): Double? = outEdges[from]?.get(to)

    fun maxEdgeWeight(): Double? {
        val allEdges = edges()
        if (allEdges.isEmpty()) return null
        var best = allEdges.first().weight
        for (edge in allEdges) {
            best = max(best, edge.weight)
        }
        return best
    }

    fun clear() {
        outEdges.clear()
        inEdges.clear()
    }

    fun isDag(): Boolean = !hasCycle()

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
            if (depth > maxDepth || results.size >= limit) return
            if (current == target) {
                results.add(path.toList())
                return
            }
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

data class Edge(
    val from: String,
    val to: String,
    val weight: Double,
)
