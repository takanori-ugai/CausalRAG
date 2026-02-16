package causalrag.examples

import causalrag.CausalRAGPipeline

fun main() {
    val documents =
        listOf(
            "Climate change causes rising sea levels, which leads to coastal flooding.",
            "Deforestation reduces carbon capture, increasing atmospheric CO2.",
            "Higher CO2 levels accelerate global warming, exacerbating climate change.",
            "Coastal flooding damages infrastructure and causes population displacement.",
            "Climate policies aim to reduce emissions, thereby mitigating climate change effects.",
        )

    val pipeline = CausalRAGPipeline(configPath = "config/causalrag.json")
    println("Pipeline initialized")

    println("Indexing documents...")
    pipeline.index(documents)
    println("Indexed ${documents.size} documents with causal relationships")

    val saveDir = "causalrag_index"
    val saved = pipeline.save(saveDir)
    if (saved) {
        println("Saved index to $saveDir")
    }

    val queries =
        listOf(
            "What are the effects of climate change?",
            "How does deforestation affect the environment?",
            "What can mitigate coastal flooding?",
        )

    for (query in queries) {
        println("\n" + "=".repeat(80))
        println("Query: $query")
        println("=".repeat(80))

        val answer = pipeline.run(query)
        println("\nAnswer: $answer")

        println("\nSupporting context:")
        val context = pipeline.hybridRetriever.retrieve(query, topK = 3).map { it as String }
        context.forEachIndexed { idx, ctx ->
            val preview = if (ctx.length > 200) ctx.substring(0, 200) + "..." else ctx
            println("[${idx + 1}] $preview")
        }

        val causalPaths = pipeline.graphRetriever.retrievePaths(query, maxPaths = 3)
        if (causalPaths.isNotEmpty()) {
            println("\nRelevant causal pathways:")
            causalPaths.forEach { path ->
                println("- ${path.joinToString(" -> ")}")
            }
        }
    }

    println("\nDemo completed!")
}
