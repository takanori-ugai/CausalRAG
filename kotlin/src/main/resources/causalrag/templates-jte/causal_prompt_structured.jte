@import java.util.List
@param String query
@param List<String> passages
@param List<List<String>> causalPaths
@param String causalGraphSummary
@param List<String> pathSummaries

You are a causal reasoning assistant that explains complex relationships between concepts.
For the following question, provide a structured answer that:
1. Identifies the key causal factors involved
2. Explains how these factors relate through causal mechanisms
3. Provides a final answer that follows from this causal chain

Use the provided context passages and causal relationship information.

@if(pathSummaries != null && !pathSummaries.isEmpty())
SUMMARY OF CAUSAL MECHANISMS: ${pathSummaries.get(0)}
@endif

@if(causalPaths != null && !causalPaths.isEmpty())
Relevant causal pathways:
@for(int i = 0; i < causalPaths.size(); i++)
[${i + 1}] ${String.join(" → ", causalPaths.get(i))}
@if(pathSummaries != null && (i + 1) < pathSummaries.size())
   Natural language: ${pathSummaries.get(i + 1)}
@endif
@endfor
@endif

@if(causalGraphSummary != null && !causalGraphSummary.isBlank())
Causal graph structure: ${causalGraphSummary}
@endif

Important: Use these causal pathways to structure your reasoning.

Context passages:
@for(int i = 0; i < passages.size(); i++)
[${i + 1}] ${passages.get(i)}
@endfor

Question: ${query}

Your structured causal answer:
1. Causal factors:
2. Causal mechanisms:
3. Conclusion:
