package causalrag.examples

import causalrag.evaluation.EvalExample
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path

object CliUtils {
    fun parseOptions(args: List<String>): Map<String, String> {
        val options = mutableMapOf<String, String>()
        var i = 0
        while (i < args.size) {
            val arg = args[i]
            if (arg.startsWith("--")) {
                val key = arg.removePrefix("--")
                val value = args.getOrNull(i + 1)
                if (value != null && !value.startsWith("-")) {
                    options[key] = value
                    i += 2
                } else {
                    options[key] = "true"
                    i += 1
                }
            } else if (arg.startsWith("-")) {
                val key = arg.removePrefix("-")
                val value = args.getOrNull(i + 1)
                if (value != null && !value.startsWith("-")) {
                    options[key] = value
                    i += 2
                } else {
                    options[key] = "true"
                    i += 1
                }
            } else {
                i += 1
            }
        }
        return options
    }

    fun loadEvaluationData(filepath: String): List<EvalExample> {
        val json = Json { ignoreUnknownKeys = true }
        val text = Files.readString(Path.of(filepath))
        val root = json.parseToJsonElement(text)
        if (root !is JsonArray) return emptyList()
        return root.mapNotNull { element ->
            val obj = element as? JsonObject ?: return@mapNotNull null
            val question = (obj["question"] as? JsonPrimitive)?.content ?: return@mapNotNull null
            val groundTruth = (obj["ground_truth"] as? JsonPrimitive)?.content
            EvalExample(question = question, groundTruth = groundTruth)
        }
    }
}
