package causalrag.reranker

import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

abstract class BaseReranker(
    val name: String = "base",
) {
    init {
        logger.debug { "Initialized $name reranker" }
    }

    abstract fun rerank(
        query: String,
        candidates: List<String>,
        metadata: List<Map<String, Any>>? = null,
    ): List<Pair<String, Double>>

    /**
     * Alias of [rerank] kept for compatibility with call sites that expect a scored rerank method.
     * This may evolve to return richer metadata in the future.
     */
    fun rerankWithScores(
        query: String,
        candidates: List<String>,
        metadata: List<Map<String, Any>>? = null,
    ): List<Pair<String, Double>> = rerank(query, candidates, metadata)

    open fun getExplanation(
        query: String,
        candidate: String,
        metadata: Map<String, Any>? = null,
    ): String = "No detailed explanation available for $name reranker."
}
