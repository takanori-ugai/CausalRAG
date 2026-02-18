package causalrag.retriever

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlin.math.ln

private val logger = KotlinLogging.logger {}

class Bm25Retriever(
    private val k1: Double = 1.5,
    private val b: Double = 0.75,
    name: String = "bm25",
) : KeywordRetriever(name) {
    private var passages: List<String> = emptyList()
    private val docFreqs: MutableMap<String, Int> = mutableMapOf()
    private val termFreqs: MutableList<Map<String, Int>> = mutableListOf()
    private val docLengths: MutableList<Int> = mutableListOf()
    private var avgDocLength: Double = 0.0
    private var numDocs: Int = 0

    override fun tokenize(text: String): List<String> = WORD_REGEX.findAll(text.lowercase()).map { it.value }.toList()

    companion object {
        private val WORD_REGEX = Regex("\\w+")
    }

    internal fun indexDocuments(texts: List<String>) {
        passages = texts
        numDocs = texts.size
        docFreqs.clear()
        termFreqs.clear()
        docLengths.clear()
        avgDocLength = 0.0

        for (doc in texts) {
            val tokens = tokenize(doc)
            val tf = mutableMapOf<String, Int>()
            for (token in tokens) {
                tf[token] = (tf[token] ?: 0) + 1
            }
            termFreqs.add(tf)
            docLengths.add(tokens.size)

            val unique = tf.keys
            for (term in unique) {
                docFreqs[term] = (docFreqs[term] ?: 0) + 1
            }
        }

        avgDocLength = if (numDocs > 0) docLengths.sum().toDouble() / numDocs else 0.0
        logger.info { "Indexed $numDocs documents for BM25" }
    }

    override fun indexCorpus(documents: List<String>): Boolean {
        if (documents.isEmpty()) return false
        indexDocuments(documents)
        return true
    }

    override fun retrieve(
        query: String,
        topK: Int,
    ): List<Map<String, Any>> {
        if (numDocs == 0) return emptyList()
        val terms = tokenize(processQuery(query))
        if (terms.isEmpty()) return emptyList()
        if (avgDocLength == 0.0) return emptyList()

        val scores = DoubleArray(numDocs)
        for (term in terms) {
            val df = docFreqs[term] ?: 0
            if (df == 0) continue
            val idf = ln((numDocs - df + 0.5) / (df + 0.5) + 1.0)
            for (docId in 0 until numDocs) {
                val tf = termFreqs[docId][term] ?: 0
                if (tf == 0) continue
                val dl = docLengths[docId].toDouble()
                val denom = tf + k1 * (1.0 - b + b * (dl / avgDocLength))
                val score = idf * (tf * (k1 + 1.0)) / denom
                scores[docId] += score
            }
        }

        val ranked =
            scores
                .mapIndexed { idx, score -> idx to score }
                .sortedByDescending { it.second }
                .take(topK)

        return ranked.mapIndexed { rank, (idx, score) ->
            mapOf(
                "passage" to passages[idx],
                "score" to score,
                "rank" to (rank + 1),
            )
        }
    }

    override fun getStats(): Map<String, Any> =
        mapOf(
            "name" to name,
            "type" to "Bm25Retriever",
            "documents" to numDocs,
            "avg_doc_length" to avgDocLength,
        )
}
