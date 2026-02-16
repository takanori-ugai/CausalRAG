package causalrag.examples

import causalrag.CausalRAGPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.DoubleAdder
import kotlin.math.max

@Serializable
data class MusiqueParagraph(
    val idx: Int,
    val title: String,
    @SerialName("paragraph_text")
    val paragraphText: String,
    @SerialName("is_supporting")
    val isSupporting: Boolean,
)

@Serializable
data class MusiqueExample(
    val id: String,
    val paragraphs: List<MusiqueParagraph>,
    val question: String,
    val answer: String,
    @SerialName("answer_aliases")
    val answerAliases: List<String> = emptyList(),
    val answerable: Boolean = true,
)

object MusiQue {
    private val inputJson = Json { ignoreUnknownKeys = true }
    private val prettyJson = Json { prettyPrint = true }

    @JvmStatic
    fun main(args: Array<String>) {
        val dataPath = Path.of("data/musique_ans_v1.0_train-200.jsonl")
        if (!Files.exists(dataPath)) {
            System.err.println("Missing MusiQue data file at $dataPath")
            kotlin.system.exitProcess(1)
        }

        val provider = (System.getenv("LLM_PROVIDER") ?: "openai").lowercase()
        val apiKey = System.getenv("OPENAI_API_KEY")
        if (provider == "openai" && apiKey.isNullOrBlank()) {
            System.err.println("OPENAI_API_KEY is required for provider=openai")
            kotlin.system.exitProcess(1)
        }

        val modelName = System.getenv("LLM_MODEL") ?: "gpt-4o-mini"
        val embeddingModel = System.getenv("EMBEDDING_MODEL") ?: "text-embedding-3-small"
        val baseUrl = System.getenv("LLM_BASE_URL")
        val limit = (System.getenv("MUSIQUE_LIMIT") ?: "").toIntOrNull()
        val parallelism = (System.getenv("MUSIQUE_PARALLELISM") ?: "5").toIntOrNull() ?: 5
        require(parallelism > 0) { "MUSIQUE_PARALLELISM must be positive." }

        val lines =
            Files
                .readAllLines(dataPath)
                .map { it.trim() }
                .filter { it.isNotBlank() }
        if (lines.isEmpty()) {
            System.err.println("No MusiQue samples found in $dataPath")
            kotlin.system.exitProcess(1)
        }

        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val resultsDir = Path.of("eval_results").resolve("musique_$timestamp")
        Files.createDirectories(resultsDir)
        val configPath = writeTempConfig(modelName, embeddingModel, provider, apiKey, baseUrl)

        val examples =
            lines
                .map { inputJson.decodeFromString(MusiqueExample.serializer(), it) }
                .filter { it.answerable }
                .let { list -> if (limit != null) list.take(limit) else list }

        if (examples.isEmpty()) {
            System.err.println("No answerable MusiQue samples were processed.")
            kotlin.system.exitProcess(1)
        }

        val exactMatchTotal = DoubleAdder()
        val f1Total = DoubleAdder()
        val processed = AtomicInteger(0)
        val semaphore = Semaphore(parallelism)

        runBlocking {
            val jobs =
                examples.map { example ->
                    async(Dispatchers.IO) {
                        semaphore.withPermit {
                            val documents = example.paragraphs.map { it.paragraphText }
                            val pipeline = CausalRAGPipeline(configPath = configPath.toString())

                            pipeline.index(documents)
                            val result = pipeline.runWithContext(example.question, topK = 5)
                            val prediction = result.answer

                            val golds = listOf(example.answer) + example.answerAliases
                            println("Answer: $prediction")
                            println("Gold: $golds")
                            val em = bestExactMatch(prediction, golds)
                            val f1 = bestF1(prediction, golds)
                            exactMatchTotal.add(em)
                            f1Total.add(f1)

                            val count = processed.incrementAndGet()
                            if ((count % 10) == 0) {
                                println("Processed $count / ${examples.size} samples")
                            }

                            val perSample =
                                buildString {
                                    appendLine("id: ${example.id}")
                                    appendLine("question: ${example.question}")
                                    appendLine("prediction: $prediction")
                                    appendLine("gold: ${example.answer}")
                                    if (example.answerAliases.isNotEmpty()) {
                                        appendLine("aliases: ${example.answerAliases.joinToString(", ")}")
                                    }
                                    appendLine("exact_match: $em")
                                    appendLine("f1: ${"%.4f".format(f1)}")
                                }
                            Files.writeString(resultsDir.resolve("${example.id}.txt"), perSample)
                        }
                    }
                }
            jobs.awaitAll()
        }

        val processedCount = processed.get()
        if (processedCount == 0) {
            System.err.println("No answerable MusiQue samples were processed.")
            kotlin.system.exitProcess(1)
        }

        val exactMatch = exactMatchTotal.sum() / processedCount
        val f1 = f1Total.sum() / processedCount
        println("MusiQue evaluation completed for $processedCount samples")
        println("Exact Match: ${"%.4f".format(exactMatch)}")
        println("F1: ${"%.4f".format(f1)}")
        println("Per-sample outputs written to $resultsDir")
    }

    private fun writeTempConfig(
        modelName: String,
        embeddingModel: String,
        provider: String,
        apiKey: String?,
        baseUrl: String?,
    ): Path {
        val tempDir = Files.createTempDirectory("causalrag-musique")
        val configPath = tempDir.resolve("musique-config.json")
        val json =
            JsonObject(
                mapOf(
                    "modelName" to JsonPrimitive(modelName),
                    "embeddingModel" to JsonPrimitive(embeddingModel),
                    "llmProvider" to JsonPrimitive(provider),
                    "llmApiKey" to (apiKey?.let { JsonPrimitive(it) } ?: JsonNull),
                    "llmBaseUrl" to (baseUrl?.let { JsonPrimitive(it) } ?: JsonNull),
                    "embeddingApiKey" to (apiKey?.let { JsonPrimitive(it) } ?: JsonNull),
                    "graphPath" to JsonNull,
                    "indexPath" to JsonNull,
                    "templateStyle" to JsonPrimitive("detailed"),
                ),
            )
        val content = prettyJson.encodeToString(JsonElement.serializer(), json)
        Files.writeString(configPath, content)
        return configPath
    }

    private fun bestExactMatch(
        prediction: String,
        golds: List<String>,
    ): Double = golds.maxOfOrNull { if (normalize(prediction) == normalize(it)) 1.0 else 0.0 } ?: 0.0

    private fun bestF1(
        prediction: String,
        golds: List<String>,
    ): Double = golds.maxOfOrNull { f1Score(prediction, it) } ?: 0.0

    private fun f1Score(
        prediction: String,
        gold: String,
    ): Double {
        val predTokens = tokenize(normalize(prediction))
        val goldTokens = tokenize(normalize(gold))
        if (predTokens.isEmpty() && goldTokens.isEmpty()) return 1.0
        if (predTokens.isEmpty() || goldTokens.isEmpty()) return 0.0

        val predCounts = predTokens.groupingBy { it }.eachCount()
        val goldCounts = goldTokens.groupingBy { it }.eachCount()
        var overlap = 0
        for ((token, pCount) in predCounts) {
            val gCount = goldCounts[token] ?: 0
            overlap += minOf(pCount, gCount)
        }
        if (overlap == 0) return 0.0
        val precision = overlap.toDouble() / predTokens.size
        val recall = overlap.toDouble() / goldTokens.size
        return 2 * precision * recall / max(precision + recall, 1e-9)
    }

    private fun normalize(text: String): String {
        val lowered = text.lowercase()
        val noPunc = lowered.replace(Regex("[^a-z0-9\\s]"), " ")
        val noArticles = noPunc.replace(Regex("\\b(a|an|the)\\b"), " ")
        return noArticles.replace(Regex("\\s+"), " ").trim()
    }

    private fun tokenize(text: String): List<String> =
        if (text.isBlank()) {
            emptyList()
        } else {
            text.split(' ')
        }
}
