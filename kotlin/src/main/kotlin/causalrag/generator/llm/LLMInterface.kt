package causalrag.generator.llm

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.ollama.OllamaChatModel
import dev.langchain4j.model.openai.OpenAiChatModel
import dev.langchain4j.model.openai.OpenAiChatRequestParameters
import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

@Suppress("TooGenericExceptionCaught")
class LLMInterface(
    private val modelName: String = "gpt-4o-mini",
    apiKey: String? = null,
    provider: String = "openai",
    private val systemMessage: String? = null,
    private val baseUrl: String? = null,
) {
    private val providerName = provider.lowercase()
    private val apiKeyValue = apiKey ?: System.getenv("OPENAI_API_KEY")

    private var chatModel: ChatModel? = null
    private var lastTemperature: Double? = null
    private var lastJsonMode: Boolean? = null
    private var lastMaxTokens: Int? = null
    private var lastJsonArrayMode: Boolean? = null

    fun generate(
        prompt: String,
        temperature: Double = 0.3,
        maxTokens: Int = 800,
        stream: Boolean = false,
        jsonMode: Boolean = false,
        jsonArrayMode: Boolean = false,
    ): String {
        if (stream) {
            logger.warn { "Streaming not implemented; falling back to non-streaming response." }
        }
        return try {
            when (providerName) {
                "openai" -> {
                    val model = getOpenAiModel(temperature, maxTokens, jsonMode, jsonArrayMode)
                    val finalPrompt =
                        if (!systemMessage.isNullOrBlank()) {
                            "${systemMessage}\n\n$prompt"
                        } else {
                            prompt
                        }
                    model.chat(finalPrompt)
                }

                "ollama" -> {
                    val model = getOllamaModel()
                    val finalPrompt =
                        if (!systemMessage.isNullOrBlank()) {
                            "${systemMessage}\n\n$prompt"
                        } else {
                            prompt
                        }
                    model.chat(finalPrompt)
                }

                else -> {
                    "Unsupported provider: $providerName"
                }
            }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error generating completion" }
            "Error generating response: ${ex.message}"
        }
    }

    private fun getOpenAiModel(
        temperature: Double,
        maxTokens: Int,
        jsonMode: Boolean,
        jsonArrayMode: Boolean,
    ): ChatModel {
        check(!apiKeyValue.isNullOrBlank()) { "OPENAI_API_KEY is not configured." }
        if (
            chatModel == null ||
            lastTemperature != temperature ||
            lastJsonMode != jsonMode ||
            lastMaxTokens != maxTokens ||
            lastJsonArrayMode != jsonArrayMode
        ) {
            val builder =
                OpenAiChatModel
                    .builder()
                    .apiKey(apiKeyValue)
                    .modelName(modelName)
                    .defaultRequestParameters(
                        OpenAiChatRequestParameters
                            .builder()
                            .modelName(modelName)
                            .temperature(temperature)
                            .maxCompletionTokens(maxTokens)
                            .build(),
                    )
            if (baseUrl != null) {
                builder.baseUrl(baseUrl)
            }
            if (jsonMode && !jsonArrayMode) {
                builder.responseFormat("json_object")
            }
            chatModel = builder.build()
            lastTemperature = temperature
            lastJsonMode = jsonMode
            lastMaxTokens = maxTokens
            lastJsonArrayMode = jsonArrayMode
        }
        return chatModel!!
    }

    private fun getOllamaModel(): ChatModel {
        if (chatModel == null) {
            val builder = OllamaChatModel.builder().modelName(modelName)
            if (baseUrl != null) {
                builder.baseUrl(baseUrl)
            }
            chatModel = builder.build()
        }
        return chatModel!!
    }
}
