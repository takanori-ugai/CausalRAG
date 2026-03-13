package causalrag.reranker

import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

/**
 * Base abstraction for rerankers that reorder retrieved passages.
 *
 * @property name Reranker name reported in diagnostics.
 */
abstract class BaseReranker(
    val name: String = "base",
) {
    init {
        logger.debug { "Initialized $name reranker" }
    }

    /**
     * Scores and orders candidate passages for a query.
     *
     * @param query User query.
     * @param candidates Candidate passages.
     * @param metadata Optional metadata aligned with [candidates].
     * @return Candidate passages paired with descending scores.
     */
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

    /**
     * Explains why a candidate received its reranking score.
     *
     * @param query User query.
     * @param candidate Candidate passage.
     * @param metadata Optional candidate metadata.
     * @return Human-readable explanation string.
     */
    open fun getExplanation(
        query: String,
        candidate: String,
        metadata: Map<String, Any>? = null,
    ): String = "No detailed explanation available for $name reranker."
}
