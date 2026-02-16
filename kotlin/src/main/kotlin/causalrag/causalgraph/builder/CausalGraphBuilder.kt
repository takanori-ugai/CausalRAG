package causalrag.causalgraph.builder

import causalrag.causalgraph.graph.DirectedGraph
import causalrag.generator.llm.LLMInterface
import causalrag.utils.EmbeddingModel
import causalrag.utils.EmbeddingModelFactory
import causalrag.utils.cosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.max

private val logger = KotlinLogging.logger {}

data class CausalTriple(
    val cause: String,
    val effect: String,
    val confidence: Double? = null,
)

@Suppress("TooGenericExceptionCaught")
class CausalTripleExtractor(
    private val method: String = "hybrid",
    private val llmInterface: LLMInterface? = null,
) {
    private val stopwords =
        setOf(
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "because",
            "by",
            "can",
            "could",
            "for",
            "from",
            "has",
            "have",
            "if",
            "in",
            "into",
            "is",
            "it",
            "its",
            "leads",
            "of",
            "on",
            "or",
            "that",
            "the",
            "then",
            "this",
            "to",
            "was",
            "were",
            "which",
            "with",
        )

    private val causalPatterns =
        listOf(
            Regex("([\\w\\s]+?)\\s+causes\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+leads to\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+results in\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("because of\\s+([\\w\\s]+?),\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+is caused by\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("if\\s+([\\w\\s]+?),\\s+then\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+contributes to\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+influences\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+leads\\s+to\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+triggers\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+induces\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
            Regex("([\\w\\s]+?)\\s+drives\\s+([\\w\\s]+)", RegexOption.IGNORE_CASE),
        )

    fun extract(text: String): List<CausalTriple> =
        when (method) {
            "rule" -> {
                ruleBasedExtraction(text)
            }

            "llm" -> {
                llmBasedExtraction(text)
            }

            else -> {
                val ruleTriples = ruleBasedExtraction(text)
                if (llmInterface != null) {
                    val llmTriples = llmBasedExtraction(text)
                    val existing = ruleTriples.map { it.cause.lowercase() + "|" + it.effect.lowercase() }.toSet()
                    val combined = ruleTriples.toMutableList()
                    for (triple in llmTriples) {
                        val key = triple.cause.lowercase() + "|" + triple.effect.lowercase()
                        if (key !in existing) {
                            combined.add(triple)
                        }
                    }
                    combined
                } else {
                    ruleTriples
                }
            }
        }

    private fun ruleBasedExtraction(text: String): List<CausalTriple> {
        val triples = mutableListOf<CausalTriple>()
        val cleanText = text.replace("\n", " ").trim()
        val sentences = cleanText.split(Regex("(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?|!)\\s"))
        for (sentence in sentences) {
            for (pattern in causalPatterns) {
                val matches = pattern.findAll(sentence)
                for (match in matches) {
                    val groups = match.groupValues
                    if (groups.size >= 3) {
                        val cause = normalizeCandidate(groups[1])
                        val effect = normalizeCandidate(groups[2])
                        if (isValidNodeText(cause) && isValidNodeText(effect)) {
                            var conf = 0.8
                            if (cause.length < 5 || effect.length < 5) {
                                conf = 0.6
                            }
                            triples.add(CausalTriple(cause, effect, conf))
                        }
                    }
                }
            }
        }
        return triples
    }

    private fun llmBasedExtraction(text: String): List<CausalTriple> {
        if (llmInterface == null) {
            logger.warn { "LLM interface not provided, cannot perform LLM-based extraction" }
            return emptyList()
        }
        val chunks = splitTextIntoChunks(text, maxLength = 3000)
        val allTriples = mutableListOf<CausalTriple>()
        for (chunk in chunks) {
            val prompt = createCausalExtractionPrompt(chunk)
            try {
                val response =
                    llmInterface.generate(
                        prompt,
                        temperature = 0.1,
                        jsonMode = true,
                        jsonArrayMode = true,
                    )
                allTriples.addAll(parseLlmResponse(response))
            } catch (ex: RuntimeException) {
                logger.error(ex) { "Error during LLM extraction" }
            }
        }
        return deduplicateTriples(allTriples)
    }

    private fun createCausalExtractionPrompt(text: String): String =
        """Extract all causal relationships from the text below as a JSON list of objects.
Each object should have 'cause', 'effect', and 'confidence' (0.0-1.0) fields.
Focus only on causal relationships, not merely correlations or temporal sequences.

TYPES OF CAUSAL RELATIONSHIPS TO IDENTIFY:
1. Direct causation: A directly causes B (e.g., "smoking causes lung cancer")
2. Mediated causation: A causes B through C (e.g., "smoking damages lung tissue, which leads to cancer")
3. Enabling causation: A creates conditions for B to occur (e.g., "drought creates conditions for wildfires")
4. Preventive causation: A prevents or reduces B (e.g., "vaccines prevent disease")
5. Contributory causation: A contributes to or increases risk of B (e.g., "pollution contributes to respiratory problems")

CONFIDENCE SCORING CRITERIA (0.0-1.0):
- 0.9-1.0: Explicit, unambiguous causal claim with strong causal language (causes, leads to, results in)
- 0.7-0.8: Clear causal relationship but with less explicit language (contributes to, influences)
- 0.5-0.6: Implied causation requiring some inference (associated with + mechanism described)
- 0.3-0.4: Suggested but uncertain causation (may cause, could lead to, is linked to)
- < 0.3: Primarily correlational or too uncertain to include

EXTRACTION GUIDELINES:
1. Focus on factual causal claims, not hypothetical or counterfactual statements
2. Extract the most specific cause and effect possible, avoiding overly general concepts
3. For complex causal chains (A→B→C→D), extract all individual links (A→B, B→C, C→D)
4. When multiple causes lead to the same effect, extract each relationship separately
5. Normalize forms but preserve key content (e.g., "global warming" vs "climate change")
6. Capture directional relationships correctly (what causes what)
7. When confidence is below 0.5, only include if the relationship is particularly significant

TEXT:
$text

OUTPUT FORMAT:
[
  {"cause": "climate change", "effect": "rising sea levels", "confidence": 0.95},
  {"cause": "rising sea levels", "effect": "coastal flooding", "confidence": 0.9},
  {"cause": "deforestation", "effect": "increased atmospheric CO2", "confidence": 0.85},
  {"cause": "regular exercise", "effect": "reduced risk of heart disease", "confidence": 0.8}
]

Ensure your response contains ONLY the valid JSON array. Do not include any other explanation or text.
CAUSAL RELATIONSHIPS:"""

    private fun parseLlmResponse(response: String): List<CausalTriple> {
        val json =
            Json {
                ignoreUnknownKeys = true
                isLenient = true
            }
        val jsonStr = response.trim()
        val startIdx = jsonStr.indexOf('[')
        val endIdx = jsonStr.lastIndexOf(']')
        if (startIdx == -1 || endIdx == -1 || endIdx <= startIdx) {
            logger.warn { "No valid JSON array found in response" }
            return emptyList()
        }
        val arrayText = jsonStr.substring(startIdx, endIdx + 1)
        return try {
            parseJsonTriples(json, arrayText)
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to parse LLM response, attempting fix-ups" }
            val fixed = fixJsonErrors(arrayText)
            try {
                parseJsonTriples(json, fixed)
            } catch (nested: RuntimeException) {
                logger.error(nested) { "Failed to parse LLM response after fix-ups" }
                emptyList()
            }
        }
    }

    private fun parseJsonTriples(
        json: Json,
        arrayText: String,
    ): List<CausalTriple> {
        val root = json.parseToJsonElement(arrayText)
        if (root !is JsonArray) return emptyList()
        val triples = mutableListOf<CausalTriple>()
        for (element in root) {
            if (element !is JsonObject) continue
            val cause = safeExtractField(element, "cause")
            val effect = safeExtractField(element, "effect")
            if (cause.isBlank() || effect.isBlank()) continue
            val confidence = safeExtractConfidence(element)
            triples.add(CausalTriple(normalizeText(cause), normalizeText(effect), confidence))
        }
        return triples
    }

    private fun fixJsonErrors(jsonStr: String): String {
        var fixed = jsonStr
        fixed = fixed.replace(Regex("'([^']*)'"), "\"$1\"")
        fixed = fixed.replace(Regex(",\\s*]"), "]")
        fixed = fixed.replace(Regex("}\\s*\\{"), "},{")
        fixed = fixed.replace(Regex("([{,]\\s*)([a-zA-Z_][a-zA-Z0-9_]*)\\s*:"), "$1\"$2\":")
        return fixed
    }

    private fun safeExtractField(
        item: JsonObject,
        field: String,
    ): String {
        item[field]?.let { return (it as? JsonPrimitive)?.content?.trim().orEmpty() }
        val alternatives =
            mapOf(
                "cause" to listOf("source", "from", "antecedent", "reason", "origin"),
                "effect" to listOf("target", "to", "consequent", "result", "destination", "outcome"),
            )
        val altList = alternatives[field] ?: emptyList()
        for (alt in altList) {
            item[alt]?.let { return (it as? JsonPrimitive)?.content?.trim().orEmpty() }
        }
        for ((key, value) in item) {
            if (key.equals(field, ignoreCase = true)) {
                return (value as? JsonPrimitive)?.content?.trim().orEmpty()
            }
        }
        return ""
    }

    private fun safeExtractConfidence(item: JsonObject): Double {
        val fields = listOf("confidence", "weight", "score", "probability", "certainty")
        for (field in fields) {
            val value = item[field]
            val number = (value as? JsonPrimitive)?.content?.toDoubleOrNull()
            if (number != null) return number.coerceIn(0.0, 1.0)
        }
        for ((key, value) in item) {
            if (key.equals("confidence", ignoreCase = true)) {
                val number = (value as? JsonPrimitive)?.content?.toDoubleOrNull()
                if (number != null) return number.coerceIn(0.0, 1.0)
            }
        }
        return 0.7
    }

    private fun normalizeText(text: String): String {
        var t = text.trim().trim('"', '\'')
        t = t.replace(Regex("\\s+"), " ")
        t = t.replace(Regex("[.,;:!?]$"), "")
        return t
    }

    private fun normalizeCandidate(text: String): String {
        var t = normalizeText(text)
        t = t.replace(Regex("^(which|that|this|these|those)\\s+", RegexOption.IGNORE_CASE), "")
        return t.trim()
    }

    private fun isValidNodeText(text: String): Boolean {
        if (text.length < 3) return false
        val tokens = text.lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }
        if (tokens.isEmpty()) return false
        if (tokens.size == 1 && stopwords.contains(tokens[0])) return false
        if (stopwords.contains(tokens.first())) return false
        return true
    }

    private fun splitTextIntoChunks(
        text: String,
        maxLength: Int,
    ): List<String> {
        if (text.length <= maxLength) return listOf(text)
        val paragraphs = text.split("\n\n")
        val chunks = mutableListOf<String>()
        var current = ""
        for (para in paragraphs) {
            if (current.length + para.length + 2 <= maxLength) {
                current = if (current.isEmpty()) para else "$current\n\n$para"
            } else {
                if (current.isNotEmpty()) chunks.add(current)
                if (para.length > maxLength) {
                    val sentences = para.split(Regex("(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?|!)\\s"))
                    var sentChunk = ""
                    for (sent in sentences) {
                        if (sentChunk.length + sent.length + 1 <= maxLength) {
                            sentChunk = if (sentChunk.isEmpty()) sent else "$sentChunk $sent"
                        } else {
                            if (sentChunk.isNotEmpty()) chunks.add(sentChunk)
                            sentChunk = sent
                        }
                    }
                    current = sentChunk
                } else {
                    current = para
                }
            }
        }
        if (current.isNotEmpty()) chunks.add(current)
        return chunks
    }

    private fun deduplicateTriples(triples: List<CausalTriple>): List<CausalTriple> {
        val unique = mutableMapOf<String, CausalTriple>()
        for (triple in triples) {
            val key = triple.cause.lowercase() + "|" + triple.effect.lowercase()
            val existing = unique[key]
            if (existing == null || (triple.confidence ?: 0.0) > (existing.confidence ?: 0.0)) {
                unique[key] = triple
            }
        }
        return unique.values.toList()
    }
}

@Suppress("TooGenericExceptionCaught")
class CausalGraphBuilder(
    modelName: String = "all-MiniLM-L6-v2",
    private val normalizeNodes: Boolean = true,
    private val confidenceThreshold: Double = 0.5,
    extractorMethod: String = "hybrid",
    llmInterface: LLMInterface? = null,
    graphPath: String? = null,
    embeddingModel: EmbeddingModel? = null,
    embeddingApiKey: String? = null,
) {
    private val graph = DirectedGraph()
    val nodeText: MutableMap<String, String> = mutableMapOf()
    private val nodeVariants: MutableMap<String, MutableList<String>> = mutableMapOf()
    val encoder: EmbeddingModel? =
        embeddingModel ?: EmbeddingModelFactory.createDefault(modelName, embeddingApiKey)
    val nodeEmbeddings: MutableMap<String, DoubleArray> = mutableMapOf()
    private val extractor = CausalTripleExtractor(method = extractorMethod, llmInterface = llmInterface)

    init {
        if (graphPath != null) {
            load(graphPath)
        }
    }

    fun addTriples(triples: List<CausalTriple>) {
        for (triple in triples) {
            val confidence = triple.confidence ?: 1.0
            if (confidence < confidenceThreshold) continue
            val causeId: String
            val effectId: String
            if (normalizeNodes) {
                causeId = getOrCreateNode(triple.cause)
                effectId = getOrCreateNode(triple.effect)
            } else {
                causeId = triple.cause
                effectId = triple.effect
                nodeText[causeId] = triple.cause
                nodeText[effectId] = triple.effect
            }
            graph.addEdge(causeId, effectId, confidence)
            if (encoder != null) {
                for (nodeId in listOf(causeId, effectId)) {
                    if (!nodeEmbeddings.containsKey(nodeId)) {
                        val text = nodeText[nodeId] ?: nodeId
                        nodeEmbeddings[nodeId] = encoder.encode(text)
                    }
                }
            }
        }
    }

    private fun getOrCreateNode(text: String): String {
        if (encoder == null) return text
        val textEmb = encoder.encode(text)
        var bestMatch: String? = null
        var bestScore = 0.0
        for ((nodeId, emb) in nodeEmbeddings) {
            val score = cosineSimilarity(textEmb, emb)
            if (score > 0.85 && score > bestScore) {
                bestMatch = nodeId
                bestScore = score
            }
        }
        if (bestMatch != null) {
            nodeVariants.getOrPut(bestMatch) { mutableListOf() }.add(text)
            return bestMatch
        }
        nodeText[text] = text
        nodeEmbeddings[text] = textEmb
        return text
    }

    fun indexDocuments(
        docs: List<String>,
        batchSize: Int = 5,
    ): Int {
        val initialEdges = graph.numberOfEdges()
        val docBatches = docs.chunked(batchSize)
        var batchCount = 0
        for (batch in docBatches) {
            batchCount += 1
            val batchTriples = mutableListOf<CausalTriple>()
            for (doc in batch) {
                if (doc.isBlank() || doc.trim().length < 10) continue
                try {
                    batchTriples.addAll(extractor.extract(doc))
                } catch (ex: RuntimeException) {
                    logger.error(ex) { "Error extracting triples from document" }
                }
            }
            addTriples(batchTriples)
            if (batchCount % 5 == 0 || batchCount == docBatches.size) {
                logger.info { "Processed batch $batchCount/${docBatches.size}: found ${batchTriples.size} causal relationships" }
            }
        }
        val newEdges = graph.numberOfEdges() - initialEdges
        logger.info { "Indexing complete: ${docs.size} documents processed" }
        logger.info { "Graph now has ${graph.numberOfNodes()} nodes and ${graph.numberOfEdges()} edges" }
        return newEdges
    }

    fun getGraph(): DirectedGraph = graph

    fun getNodeVariants(nodeId: String): List<String> {
        val variants = nodeVariants[nodeId] ?: emptyList()
        return listOfNotNull(nodeText[nodeId]) + variants
    }

    fun getEmbedding(nodeId: String): DoubleArray? = nodeEmbeddings[nodeId]

    fun describeGraph(): String {
        if (graph.numberOfEdges() == 0) return "Empty causal graph (no causal relationships found)"
        return graph.edges().joinToString("\n") { edge ->
            val aText = nodeText[edge.from] ?: edge.from
            val bText = nodeText[edge.to] ?: edge.to
            "$aText -> $bText (confidence: ${"%.2f".format(edge.weight)})"
        }
    }

    fun save(filepath: String) {
        val json = Json { prettyPrint = true }
        val nodeJson =
            buildJsonObject {
                for ((k, v) in nodeText) {
                    put(k, v)
                }
            }
        val variantsJson =
            buildJsonObject {
                for ((k, v) in nodeVariants) {
                    putJsonArray(k) {
                        v.forEach { add(JsonPrimitive(it)) }
                    }
                }
            }
        val edgesJson =
            buildJsonObject {
                putJsonArray("edges") {
                    for (edge in graph.edges()) {
                        add(
                            buildJsonObject {
                                put("from", edge.from)
                                put("to", edge.to)
                                put("weight", edge.weight)
                            },
                        )
                    }
                }
            }
        val root =
            buildJsonObject {
                put("nodes", nodeJson)
                put("variants", variantsJson)
                put("edges", edgesJson["edges"] ?: JsonArray(emptyList()))
            }
        Files.writeString(Path.of(filepath), json.encodeToString(JsonElement.serializer(), root))
    }

    fun load(filepath: String): Boolean {
        val path = Path.of(filepath)
        if (!Files.exists(path)) {
            logger.error { "Graph file not found: $filepath" }
            return false
        }
        return try {
            val json = Json { ignoreUnknownKeys = true }
            val root = json.parseToJsonElement(Files.readString(path)) as? JsonObject ?: return false
            val nodesObj = root["nodes"] as? JsonObject ?: JsonObject(emptyMap())
            val variantsObj = root["variants"] as? JsonObject ?: JsonObject(emptyMap())
            val edgesArray = root["edges"] as? JsonArray ?: JsonArray(emptyList())

            graph.clear()
            nodeText.clear()
            nodeVariants.clear()
            nodeEmbeddings.clear()

            for ((k, v) in nodesObj) {
                nodeText[k] = (v as? JsonPrimitive)?.content ?: k
            }
            for ((k, v) in variantsObj) {
                val list = mutableListOf<String>()
                if (v is JsonArray) {
                    for (item in v) {
                        list.add((item as? JsonPrimitive)?.content.orEmpty())
                    }
                }
                nodeVariants[k] = list
            }

            for (edge in edgesArray) {
                if (edge !is JsonObject) continue
                val from = (edge["from"] as? JsonPrimitive)?.content ?: continue
                val to = (edge["to"] as? JsonPrimitive)?.content ?: continue
                val weight = (edge["weight"] as? JsonPrimitive)?.content?.toDoubleOrNull() ?: 1.0
                graph.addEdge(from, to, weight)
            }

            if (encoder != null) {
                for ((nodeId, text) in nodeText) {
                    nodeEmbeddings[nodeId] = encoder.encode(text)
                }
            }
            true
        } catch (ex: IOException) {
            logger.error(ex) { "Error loading graph from $filepath" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error loading graph from $filepath" }
            false
        }
    }

    fun getExtractionStatistics(): Map<String, Any> {
        if (graph.numberOfEdges() == 0) return mapOf("error" to "Graph is empty")
        val nodes = graph.numberOfNodes()
        val edges = graph.numberOfEdges()
        val inDegrees = graph.nodes().map { graph.inDegree(it) }
        val outDegrees = graph.nodes().map { graph.outDegree(it) }

        val edgeConfidences = graph.edges().sortedByDescending { it.weight }
        val topRelationships =
            edgeConfidences.take(10).map { edge ->
                mapOf(
                    "cause" to (nodeText[edge.from] ?: edge.from),
                    "effect" to (nodeText[edge.to] ?: edge.to),
                    "confidence" to edge.weight,
                )
            }

        val centralNodes =
            graph
                .nodes()
                .map { node ->
                    node to (graph.inDegree(node) + graph.outDegree(node))
                }.sortedByDescending { it.second }

        val centralConcepts =
            centralNodes.take(10).map { (node, degree) ->
                mapOf(
                    "concept" to (nodeText[node] ?: node),
                    "connections" to degree,
                    "in_degree" to graph.inDegree(node),
                    "out_degree" to graph.outDegree(node),
                )
            }

        val density = if (nodes > 1) edges.toDouble() / (nodes * (nodes - 1)) else 0.0

        val isDag = graph.isDag()
        val hasCycles = graph.hasCycle()
        val components = graph.weaklyConnectedComponents().size

        return mapOf(
            "graph_statistics" to
                mapOf(
                    "nodes" to nodes,
                    "edges" to edges,
                    "density" to density,
                    "is_dag" to isDag,
                    "has_cycles" to hasCycles,
                    "components" to components,
                ),
            "degree_statistics" to
                mapOf(
                    "max_in_degree" to (inDegrees.maxOrNull() ?: 0),
                    "avg_in_degree" to if (inDegrees.isNotEmpty()) inDegrees.average() else 0.0,
                    "max_out_degree" to (outDegrees.maxOrNull() ?: 0),
                    "avg_out_degree" to if (outDegrees.isNotEmpty()) outDegrees.average() else 0.0,
                ),
            "top_relationships" to topRelationships,
            "central_concepts" to centralConcepts,
        )
    }

    fun visualizeGraph(
        outputPath: String? = null,
        format: String = "json",
        maxNodes: Int = 100,
        minEdgeWeight: Double = 0.0,
        highlightNodes: List<String>? = null,
        title: String = "Causal Knowledge Graph",
    ): String? {
        if (graph.numberOfNodes() == 0) {
            logger.error { "Cannot visualize empty graph" }
            return null
        }

        var graphToViz: DirectedGraph = graph
        if (graph.numberOfNodes() > maxNodes) {
            val nodeImportance = mutableMapOf<String, Double>()
            for (node in graph.nodes()) {
                var totalWeight = 0.0
                for (edge in graph.edges()) {
                    if (edge.from == node || edge.to == node) {
                        totalWeight += edge.weight
                    }
                }
                nodeImportance[node] = totalWeight
            }
            val importantNodes =
                nodeImportance.entries
                    .sortedByDescending { it.value }
                    .take(maxNodes)
                    .map { it.key }
            graphToViz = graph.subgraph(importantNodes.toSet())
        }

        val filteredEdges = graphToViz.edges().filter { it.weight >= minEdgeWeight }
        val nodeSet = filteredEdges.flatMap { listOf(it.from, it.to) }.toSet()
        val filteredGraph = graphToViz.subgraph(nodeSet)

        return if (format.lowercase() == "json") {
            val json = Json { prettyPrint = true }
            val nodesJson =
                filteredGraph.nodes().map { node ->
                    buildJsonObject {
                        put("id", node)
                        put("label", nodeText[node] ?: node)
                        put("in_degree", filteredGraph.inDegree(node))
                        put("out_degree", filteredGraph.outDegree(node))
                        put("highlighted", highlightNodes?.contains(node) ?: false)
                    }
                }
            val edgesJson =
                filteredEdges.map { edge ->
                    buildJsonObject {
                        put("source", edge.from)
                        put("target", edge.to)
                        put("weight", edge.weight)
                    }
                }
            val root =
                buildJsonObject {
                    putJsonArray("nodes") { nodesJson.forEach { add(it) } }
                    putJsonArray("edges") { edgesJson.forEach { add(it) } }
                    putJsonArray("metadata") {
                        add(buildJsonObject { put("title", title) })
                    }
                }
            val content = json.encodeToString(JsonElement.serializer(), root)
            if (outputPath != null) {
                Files.writeString(Path.of(outputPath), content)
                outputPath
            } else {
                content
            }
        } else {
            logger.warn { "Unsupported visualization format: $format" }
            null
        }
    }
}
