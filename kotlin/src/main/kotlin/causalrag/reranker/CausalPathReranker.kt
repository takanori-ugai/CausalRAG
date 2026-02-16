package causalrag.reranker

import causalrag.causalgraph.retriever.CausalPathRetriever
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

@Suppress("TooGenericExceptionCaught")
class CausalPathReranker(
    private val retriever: CausalPathRetriever,
    name: String = "causal_path",
    private val nodeMatchWeight: Double = 1.0,
    private val pathMatchWeight: Double = 2.0,
    private val semanticMatchWeight: Double = 0.5,
    private val minNodeLength: Int = 3,
) : BaseReranker(name) {
    override fun rerank(
        query: String,
        candidates: List<String>,
        metadata: List<Map<String, Any>>?,
    ): List<Pair<String, Double>> {
        if (candidates.isEmpty()) return emptyList()

        val pathNodes: List<String>
        val causalPaths: List<List<String>>
        try {
            pathNodes = retriever.retrievePathNodes(query)
            causalPaths = retriever.retrievePaths(query, maxPaths = 3)
            if (pathNodes.isEmpty()) {
                logger.warn { "No causal nodes found for query: $query" }
                return candidates.map { it to 0.1 }
            }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error retrieving causal information" }
            return candidates.map { it to 0.1 }
        }

        val scoredCandidates = mutableListOf<Pair<String, Double>>()
        for ((index, passage) in candidates.withIndex()) {
            val nodeScore = calculateNodeOverlap(passage, pathNodes)
            val pathScore = calculatePathStructure(passage, causalPaths)
            val semanticScore = metadata?.getOrNull(index)?.get("score") as? Double ?: 0.0
            val finalScore =
                nodeScore * nodeMatchWeight +
                    pathScore * pathMatchWeight +
                    semanticScore * semanticMatchWeight
            scoredCandidates.add(passage to finalScore)
        }

        val scores = scoredCandidates.map { it.second }
        val maxScore = scores.maxOrNull() ?: 1.0
        val minScore = scores.minOrNull() ?: 0.0
        val range = maxOf(maxScore - minScore, 1e-5)
        val normalized = scoredCandidates.map { it.first to ((it.second - minScore) / range) }
        return normalized.sortedByDescending { it.second }
    }

    private fun calculateNodeOverlap(
        passage: String,
        nodes: List<String>,
    ): Double {
        if (nodes.isEmpty()) return 0.0
        val passageLower = passage.lowercase()
        val eligibleNodes = nodes.filter { it.length >= minNodeLength }
        if (eligibleNodes.isEmpty()) return 0.0
        var matchCount = 0
        for (node in eligibleNodes) {
            if (passageLower.contains(node.lowercase())) {
                matchCount += 1
            }
        }
        return matchCount.toDouble() / eligibleNodes.size.toDouble()
    }

    private fun calculatePathStructure(
        passage: String,
        paths: List<List<String>>,
    ): Double {
        if (paths.isEmpty()) return 0.0
        val passageLower = passage.lowercase()
        val pathScores = mutableListOf<Double>()
        for (path in paths) {
            if (path.size < 2) continue
            var pairMatches = 0.0
            var totalPairs = 0
            for (i in 0 until path.size - 1) {
                val cause = path[i].lowercase()
                val effect = path[i + 1].lowercase()
                if (cause.length < minNodeLength || effect.length < minNodeLength) continue
                totalPairs += 1
                if (passageLower.contains(cause) && passageLower.contains(effect)) {
                    val causePos = passageLower.indexOf(cause)
                    val effectPos = passageLower.indexOf(effect)
                    pairMatches += if (causePos < effectPos) 1.5 else 0.5
                }
            }
            if (totalPairs > 0) {
                pathScores.add(pairMatches / totalPairs.toDouble())
            }
        }
        return if (pathScores.isNotEmpty()) pathScores.average() else 0.0
    }

    override fun getExplanation(
        query: String,
        candidate: String,
        metadata: Map<String, Any>?,
    ): String {
        val (pathNodes, causalPaths) = getQueryContext(query)
        val nodeMatches =
            pathNodes.filter { node ->
                node.length >= minNodeLength && candidate.lowercase().contains(node.lowercase())
            }
        val pathMatches = mutableListOf<String>()
        for (path in causalPaths) {
            if (path.size < 2) continue
            for (i in 0 until path.size - 1) {
                val cause = path[i].lowercase()
                val effect = path[i + 1].lowercase()
                if (
                    cause.length >= minNodeLength &&
                    effect.length >= minNodeLength &&
                    candidate.lowercase().contains(cause) &&
                    candidate.lowercase().contains(effect)
                ) {
                    pathMatches.add("$cause -> $effect")
                }
            }
        }

        val explanation = mutableListOf("Causal ranking explanation for passage:")
        if (nodeMatches.isNotEmpty()) {
            explanation.add("\nMatched concepts (${nodeMatches.size}/${pathNodes.size}):")
            nodeMatches.take(5).forEach { explanation.add("- $it") }
            if (nodeMatches.size > 5) {
                explanation.add("- ... and ${nodeMatches.size - 5} more")
            }
        } else {
            explanation.add("\nNo direct concept matches found.")
        }

        if (pathMatches.isNotEmpty()) {
            explanation.add("\nPreserved causal relationships:")
            pathMatches.take(3).forEach { explanation.add("- $it") }
            if (pathMatches.size > 3) {
                explanation.add("- ... and ${pathMatches.size - 3} more")
            }
        } else {
            explanation.add("\nNo causal relationships preserved.")
        }

        return explanation.joinToString("\n")
    }

    private fun getQueryContext(query: String): Pair<List<String>, List<List<String>>> =
        try {
            val nodes = retriever.retrievePathNodes(query)
            val paths = retriever.retrievePaths(query, maxPaths = 3)
            nodes to paths
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error retrieving causal information for explanation" }
            emptyList<String>() to emptyList()
        }
}
