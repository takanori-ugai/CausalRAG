# CausalRAG Kotlin

This module is the Kotlin port of the Python project described in `../README.MD`. It preserves the same pipeline concepts and prompt templates while adapting the build, configuration, and execution flow to the JVM.

## Key Features
- Causal triple extraction and graph construction
- Hybrid retrieval with semantic, BM25, and causal signals
- Graph-aware reranking
- Prompt templating with JTE
- Evaluation pipeline and CLI utilities

## Requirements
- JDK 21+
- Gradle (wrapper included)
- `OPENAI_API_KEY` for OpenAI-backed models

## Getting Started

Build:
```bash
./gradlew build
```

Run the basic example:
```bash
./gradlew execute -PmainClass=causalrag.examples.BasicUsageKt
```

Run the CLI:
```bash
./gradlew execute -PmainClass=causalrag.examples.CliKt --args="--help"
```

## Configuration

The Kotlin pipeline supports a JSON config that mirrors the Python settings. A sample config is provided at `config/causalrag.json`.

Example:
```json
{
  "modelName": "gpt-4o-mini",
  "embeddingModel": "text-embedding-3-small",
  "llmProvider": "openai",
  "llmApiKey": null,
  "llmBaseUrl": null,
  "embeddingApiKey": null,
  "graphPath": null,
  "indexPath": null,
  "templateStyle": "detailed"
}
```

Usage from code:
```kotlin
val pipeline = CausalRAGPipeline(configPath = "config/causalrag.json")
pipeline.index(documents)
val answer = pipeline.run("What causes coastal flooding?")
```

Example usage (Kotlin):
```kotlin
val pipeline = CausalRAGPipeline()

// Index your documents (builds both vector index and causal graph)
val documents =
    listOf(
        "Climate change causes rising sea levels, which leads to coastal flooding.",
        "Deforestation reduces carbon capture, increasing atmospheric CO2.",
        "Higher CO2 levels accelerate global warming, exacerbating climate change.",
        "Coastal flooding damages infrastructure and causes population displacement.",
        "Climate policies aim to reduce emissions, thereby mitigating climate change effects.",
    )
pipeline.index(documents)

// Query the system
val answer = pipeline.run("What are the consequences of climate change?")
println(answer)

// Access supporting information
println("\nRelevant causal paths:")
pipeline.graphRetriever.retrievePaths("What are the consequences of climate change?", maxPaths = 3)
    .forEach { path ->
        println(path.joinToString(" -> "))
    }

println("\nSupporting contexts:")
pipeline.hybridRetriever.retrieve("What are the consequences of climate change?", topK = 5)
    .map { it as String }
    .forEach { ctx ->
        val preview = if (ctx.length > 100) ctx.substring(0, 100) + "..." else ctx
        println("- $preview")
    }
```

## Command Line Interface

The Kotlin CLI mirrors the Python CLI structure.

Index documents:
```bash
./gradlew execute -PmainClass=causalrag.examples.CliKt --args="index --input docs/ --output causalrag_index"
```

Query an existing index:
```bash
./gradlew execute -PmainClass=causalrag.examples.CliKt --args="query --index causalrag_index --query \"What causes coastal flooding?\""
```

Evaluate a pipeline:
```bash
./gradlew execute -PmainClass=causalrag.examples.CliKt --args="evaluate --eval-data ../data/evaluation/evaluation_dataset.json --index causalrag_index"
```

Note: `serve` is not implemented in the Kotlin port yet.

## Prompt Templates

Kotlin uses JTE templates stored in `src/main/resources/causalrag/templates-jte`. The template style is selected via `templateStyle` in the config:
- `default`
- `detailed`
- `structured`
- `chain_of_thought`

## Project Structure

```text
src/main/kotlin/causalrag/
  Pipeline.kt
  causalgraph/
  evaluation/
  generator/
  retriever/
  reranker/
  utils/
src/main/resources/causalrag/templates-jte/
config/
examples/
```

## Integration Options

Standalone QA System (Kotlin):
```kotlin
val pipeline = CausalRAGPipeline()
pipeline.index(documents)
val result = pipeline.run("What causes coastal flooding?")
println(result)
```

Plugin for Existing RAG (Kotlin):
```kotlin
val builder = CausalGraphBuilder()
builder.indexDocuments(documents)

val retriever = CausalPathRetriever(builder)
val reranker = CausalPathReranker(retriever)

// Use in your existing RAG pipeline
val candidates = yourExistingRetriever.retrieve(query)
val rerankedCandidates = reranker.rerank(query, candidates)
```

## Evaluation

The evaluation pipeline supports causal consistency, causal completeness, and answer quality metrics. Results are written to `./eval_results` by default.

Example:
```bash
./gradlew execute -PmainClass=causalrag.examples.EvaluatePipelineKt --args="--eval-data ../data/evaluation/evaluation_dataset.json --index causalrag_index"
```

Programmatic evaluation example (Kotlin):
```kotlin
val pipeline = CausalRAGPipeline(modelName = "gpt-4")

// Load your evaluation data
val evalData =
    listOf(
        EvalExample(
            question = "How does climate change lead to coastal flooding?",
            groundTruth = "Climate change causes rising sea levels through melting ice caps...",
        ),
        // More evaluation examples...
    )

// Run evaluation
val results =
    CausalEvaluator.evaluatePipeline(
        pipeline = pipeline,
        evalData = evalData,
        metrics = listOf("faithfulness", "causal_consistency"),
        llmInterface = LLMInterface(modelName = "gpt-4"),
        resultsDir = "./results/evaluation",
    )

// Print results
results.metrics.forEach { (metric, score) ->
    println("$metric: ${"%.4f".format(score)}")
}
```

## License

MIT License. See `LICENSE`.
