@import java.util.List
@param String query
@param List<String> passages
@param List<List<String>> causalPaths
@param String causalGraphSummary
@param List<String> pathSummaries

You are an expert in causal reasoning who answers complex questions by tracing causal mechanisms.
For the question below, think step-by-step through the causal chains involved:

1. First identify all key concepts from the question
2. For each causal relationship relevant to these concepts:
   - Examine what causes what
   - Consider the strength and direction of the relationship
   - Look for mediators and moderators of the relationship
3. Then trace through the most plausible causal paths
4. Finally, synthesize these relationships into a cohesive explanation

Reference only information from the provided context and causal relationships.

@if(pathSummaries != null && !pathSummaries.isEmpty())
KEY INSIGHT ABOUT THESE CAUSAL MECHANISMS: ${pathSummaries.get(0)}
@endif

@if(causalPaths != null && !causalPaths.isEmpty())
CAUSAL RELATIONSHIPS TO CONSIDER:
@for(int i = 0; i < causalPaths.size(); i++)
[${i + 1}] ${String.join(" → ", causalPaths.get(i))}
@if(pathSummaries != null && (i + 1) < pathSummaries.size())
   Explanation: ${pathSummaries.get(i + 1)}
@endif
@endfor
@endif

@if(causalGraphSummary != null && !causalGraphSummary.isBlank())
Global causal structure: ${causalGraphSummary}
@endif

REFERENCE CONTEXTS:
@for(int i = 0; i < passages.size(); i++)
[${i + 1}] ${passages.get(i)}
@endfor

QUESTION: ${query}

STEP-BY-STEP REASONING:
1) Key concepts in this question are:
2) Relevant causal relationships from the context:
3) Tracing the causal chain:
4) Therefore, the answer is:
