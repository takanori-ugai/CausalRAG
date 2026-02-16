package causalrag.retriever

import causalrag.causalgraph.retriever.CausalPathRetriever
import io.github.oshai.kotlinlogging.KotlinLogging
import java.util.Collections
import java.util.LinkedHashMap

private val logger = KotlinLogging.logger {}

@Suppress("TooGenericExceptionCaught")
class HybridRetriever(
    private val vectorRetriever: VectorStoreRetriever,
    private val graphRetriever: CausalPathRetriever,
    private var semanticWeight: Double = 0.4,
    private var causalWeight: Double = 0.6,
    private var bm25Weight: Double = 0.0,
    private val bm25Retriever: Bm25Retriever? = null,
    private val rerankingFactor: Int = 2,
    private val minCausalMatches: Int = 1,
    private val cacheResults: Boolean = true,
    private val cacheMaxEntries: Int = 1000,
) {
    private val queryCache: MutableMap<String, List<Map<String, Any>>> =
        if (cacheResults) {
            Collections.synchronizedMap(
                object : LinkedHashMap<String, List<Map<String, Any>>>(16, 0.75f, true) {
                    override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, List<Map<String, Any>>>?): Boolean =
                        size > cacheMaxEntries
                },
            )
        } else {
            mutableMapOf()
        }

    init {
        if (cacheResults) {
            require(cacheMaxEntries > 0) { "cacheMaxEntries must be positive when cacheResults is enabled." }
        }
        val total = semanticWeight + causalWeight + bm25Weight
        require(total > 0.0) { "Sum of semanticWeight, causalWeight, and bm25Weight must be positive." }
        if (kotlin.math.abs(total - 1.0) > 1e-9) {
            semanticWeight /= total
            causalWeight /= total
            if (bm25Weight > 0.0) {
                bm25Weight /= total
            }
        }
    }

    private fun scorePassage(
        passage: String,
        pathNodes: List<String>,
        causalPaths: List<List<String>>,
        semanticScore: Double = 0.0,
        bm25Score: Double = 0.0,
    ): Pair<Double, Map<String, Any>> {
        val passageLower = passage.lowercase()
        val matchedNodes = pathNodes.filter { passageLower.contains(it.lowercase()) }
        val nodeMatchScore = matchedNodes.size.toDouble() / maxOf(pathNodes.size, 1)

        val pathMatches = mutableListOf<Pair<String, String>>()
        var totalPairs = 0
        for (path in causalPaths) {
            if (path.size < 2) continue
            for (i in 0 until path.size - 1) {
                val cause = path[i].lowercase()
                val effect = path[i + 1].lowercase()
                totalPairs += 1
                if (passageLower.contains(cause) && passageLower.contains(effect)) {
                    val causePos = passageLower.indexOf(cause)
                    val effectPos = passageLower.indexOf(effect)
                    // Heuristic assumes cause appears before effect; reversed phrasing may be missed.
                    if (causePos < effectPos) {
                        pathMatches.add(cause to effect)
                    }
                }
            }
        }
        val overallPathScore = if (totalPairs > 0) pathMatches.size.toDouble() / totalPairs else 0.0
        val causalScore = 0.7 * nodeMatchScore + 0.3 * overallPathScore
        val combinedScore = semanticWeight * semanticScore + causalWeight * causalScore
        val finalScore = combinedScore + bm25Weight * bm25Score

        return finalScore to
            mapOf(
                "matched_nodes" to matchedNodes,
                "node_score" to nodeMatchScore,
                "path_matches" to pathMatches,
                "path_score" to overallPathScore,
                "causal_score" to causalScore,
                "semantic_score" to semanticScore,
                "bm25_score" to bm25Score,
                "combined_score" to finalScore,
            )
    }

    fun retrieve(
        query: String,
        topK: Int = 5,
    ): List<String> = retrieveWithDetails(query, topK).map { it["passage"] as String }

    fun retrieveWithScores(
        query: String,
        topK: Int = 5,
    ): List<Pair<String, Double>> = retrieveWithDetails(query, topK).map { it["passage"] as String to (it["score"] as Double) }

    fun retrieveWithDetails(
        query: String,
        topK: Int = 5,
    ): List<Map<String, Any>> {
        if (cacheResults) {
            val cached = queryCache[query]
            if (cached != null) {
                return cached.take(topK)
            }
        }

        val expandedK = topK * rerankingFactor
        val semanticResults =
            try {
                vectorRetriever.searchWithScores(query, topK = expandedK)
            } catch (ex: RuntimeException) {
                logger.error(ex) { "Error retrieving vector results" }
                emptyList()
            }

        val bm25Scores: Map<String, Double> =
            if (bm25Retriever != null && bm25Weight > 0.0) {
                try {
                    val bm25Results = bm25Retriever.retrieve(query, topK = expandedK)
                    bm25Results.associate { result ->
                        val passage = result["passage"] as String
                        val score = result["score"] as Double
                        passage to score
                    }
                } catch (ex: RuntimeException) {
                    logger.error(ex) { "Error retrieving BM25 results" }
                    emptyMap()
                }
            } else {
                emptyMap()
            }

        if (semanticResults.isEmpty()) {
            logger.warn { "No semantic results found for query" }
            return emptyList()
        }

        val pathNodes =
            try {
                graphRetriever.retrievePathNodes(query)
            } catch (ex: RuntimeException) {
                logger.error(ex) { "Error retrieving causal nodes" }
                emptyList()
            }
        val causalPaths =
            try {
                graphRetriever.retrievePaths(query, maxPaths = 3)
            } catch (ex: RuntimeException) {
                logger.error(ex) { "Error retrieving causal paths" }
                emptyList()
            }

        val scoredResults = mutableListOf<Map<String, Any>>()
        for ((passage, semanticScore) in semanticResults) {
            val keywordScore = bm25Scores[passage] ?: 0.0
            val (score, details) = scorePassage(passage, pathNodes, causalPaths, semanticScore, keywordScore)
            val matchedNodes = details["matched_nodes"] as? List<*> ?: emptyList<Any>()
            if (matchedNodes.size < minCausalMatches && pathNodes.isNotEmpty()) {
                continue
            }
            scoredResults.add(
                mapOf(
                    "passage" to passage,
                    "score" to score,
                    "details" to details,
                ),
            )
        }

        val sorted = scoredResults.sortedByDescending { it["score"] as Double }
        if (cacheResults) {
            queryCache[query] = sorted
        }
        return sorted.take(topK)
    }

    fun getExplanation(
        query: String,
        passage: String,
    ): String {
        val fallbackMessage =
            "This passage was retrieved as relevant to the query: $query. " +
                "No detailed scoring information is available."
        val cached = queryCache[query] ?: return fallbackMessage
        val result =
            cached.firstOrNull { it["passage"] == passage }
                ?: return fallbackMessage
        val details = result["details"] as Map<*, *>
        val explanation = mutableListOf("Hybrid retrieval explanation for: $query")
        val semanticScore = details["semantic_score"] as? Double ?: 0.0
        val causalScore = details["causal_score"] as? Double ?: 0.0
        val combinedScore = details["combined_score"] as? Double ?: 0.0
        val bm25Score = details["bm25_score"] as? Double ?: 0.0
        explanation.add("\nSemantic relevance score: ${"%.2f".format(semanticScore)} (weight: ${"%.2f".format(semanticWeight)})")
        explanation.add("\nCausal relevance score: ${"%.2f".format(causalScore)} (weight: ${"%.2f".format(causalWeight)})")
        if (bm25Weight > 0.0) {
            explanation.add("\nBM25 relevance score: ${"%.2f".format(bm25Score)} (weight: ${"%.2f".format(bm25Weight)})")
        }
        val matchedNodes = details["matched_nodes"] as? List<*> ?: emptyList<Any>()
        if (matchedNodes.isNotEmpty()) {
            explanation.add("\nMatched causal concepts (${matchedNodes.size} concepts):")
            matchedNodes.forEach { explanation.add("- $it") }
        }
        val pathMatches = details["path_matches"] as? List<*> ?: emptyList<Any>()
        if (pathMatches.isNotEmpty()) {
            explanation.add("\nPreserved causal relationships:")
            pathMatches.forEach { explanation.add("- $it") }
        }
        explanation.add("\nOverall score: ${"%.2f".format(combinedScore)}")
        return explanation.joinToString("\n")
    }

    fun clearCache() {
        queryCache.clear()
    }
}
