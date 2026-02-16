package causalrag.causalgraph.retriever

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.graph.DirectedGraph
import causalrag.utils.EmbeddingModel
import causalrag.utils.cosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

@Suppress("TooGenericExceptionCaught")
class CausalPathRetriever(
    private val builder: CausalGraphBuilder,
) {
    private val graph: DirectedGraph = builder.getGraph()
    private val nodeEmbeddings = builder.nodeEmbeddings
    private val nodeText = builder.nodeText
    private val encoder: EmbeddingModel? = builder.getEncoder()

    fun retrieveNodes(
        query: String,
        topK: Int = 5,
        threshold: Double = 0.5,
    ): List<Pair<String, Double>> {
        if (encoder == null || nodeEmbeddings.isEmpty()) {
            logger.warn { "Encoder or node embeddings not available" }
            return emptyList()
        }
        return try {
            val qEmb = encoder.encode(query)
            val scores = mutableMapOf<String, Double>()
            for ((nodeId, emb) in nodeEmbeddings) {
                val sim = cosineSimilarity(qEmb, emb)
                if (sim >= threshold) {
                    scores[nodeId] = sim
                }
            }
            scores.entries
                .sortedByDescending { it.value }
                .take(topK)
                .map { it.key to it.value }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error retrieving nodes" }
            emptyList()
        }
    }

    fun retrievePathNodes(
        query: String,
        topK: Int = 5,
        maxHops: Int = 2,
        includeSimilar: Boolean = true,
    ): List<String> {
        val topNodes = retrieveNodes(query, topK = topK)
        val pathNodes = mutableSetOf<String>()
        val seedNodes = topNodes.map { it.first }

        for (nodeId in seedNodes) {
            pathNodes.add(nodeId)
            pathNodes.addAll(getDescendants(nodeId, maxHops))
            pathNodes.addAll(getAncestors(nodeId, maxHops))
        }

        if (includeSimilar && seedNodes.isNotEmpty() && encoder != null) {
            try {
                val seedEmbeddings = seedNodes.mapNotNull { nodeEmbeddings[it] }
                if (seedEmbeddings.isNotEmpty()) {
                    val avgEmb = averageEmbedding(seedEmbeddings)
                    for ((nodeId, emb) in nodeEmbeddings) {
                        if (nodeId !in pathNodes) {
                            val sim = cosineSimilarity(avgEmb, emb)
                            if (sim > 0.8) {
                                pathNodes.add(nodeId)
                            }
                        }
                    }
                }
            } catch (ex: RuntimeException) {
                logger.error(ex) { "Error including similar nodes" }
            }
        }

        return pathNodes.toList()
    }

    private fun averageEmbedding(vectors: List<DoubleArray>): DoubleArray {
        val size = vectors.firstOrNull()?.size ?: 0
        val avg = DoubleArray(size)
        for (vec in vectors) {
            for (i in vec.indices) {
                avg[i] += vec[i]
            }
        }
        for (i in avg.indices) {
            avg[i] /= vectors.size.toDouble()
        }
        return avg
    }

    private fun getDescendants(
        node: String,
        maxHops: Int,
    ): Set<String> {
        val descendants = mutableSetOf<String>()
        var current = setOf(node)
        repeat(maxHops) {
            val next = mutableSetOf<String>()
            for (n in current) {
                next.addAll(graph.successors(n))
            }
            descendants.addAll(next)
            current = next
            if (current.isEmpty()) return@repeat
        }
        return descendants
    }

    private fun getAncestors(
        node: String,
        maxHops: Int,
    ): Set<String> {
        val ancestors = mutableSetOf<String>()
        var current = setOf(node)
        repeat(maxHops) {
            val next = mutableSetOf<String>()
            for (n in current) {
                next.addAll(graph.predecessors(n))
            }
            ancestors.addAll(next)
            current = next
            if (current.isEmpty()) return@repeat
        }
        return ancestors
    }

    fun retrievePaths(
        query: String,
        maxPaths: Int = 5,
        minPathLength: Int = 2,
        maxPathLength: Int = 4,
    ): List<List<String>> {
        val relevantNodes = retrievePathNodes(query, topK = 5, maxHops = 1)
        if (relevantNodes.size < 2) return emptyList()

        val paths = mutableListOf<Pair<List<String>, List<String>>>()
        val maxTotal = maxPaths * 3
        for (i in relevantNodes.indices) {
            for (j in i + 1 until relevantNodes.size) {
                if (paths.size >= maxTotal) break
                val src = relevantNodes[i]
                val tgt = relevantNodes[j]
                if (src == tgt) continue
                for ((start, end) in listOf(src to tgt, tgt to src)) {
                    if (paths.size >= maxTotal) break
                    val found = graph.findPaths(start, end, maxPathLength, limit = maxTotal - paths.size)
                    for (path in found) {
                        if (path.size >= minPathLength) {
                            val textPath = path.map { nodeText[it] ?: it }
                            paths.add(path to textPath)
                            if (paths.size >= maxTotal) break
                        }
                    }
                }
            }
        }

        val sorted = paths.sortedBy { it.first.size }
        return sorted.take(maxPaths).map { it.second }
    }

    fun getCausalExplanation(query: String): String {
        val paths = retrievePaths(query, maxPaths = 3)
        if (paths.isEmpty()) return "No relevant causal relationships found."
        val builder = StringBuilder()
        builder.append("Causal relationships relevant to '").append(query).append("':\n\n")
        for ((idx, path) in paths.withIndex()) {
            builder
                .append(idx + 1)
                .append(". ")
                .append(path.joinToString(" -> "))
                .append("\n")
        }
        return builder.toString()
    }

    fun highlightSubgraph(query: String): DirectedGraph {
        val nodes = retrievePathNodes(query)
        return graph.subgraph(nodes.toSet())
    }
}
