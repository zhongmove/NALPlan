# NALPlan

**LLM-Based Multi-Path Query Plan Synthesis for Natural Language Queries of Spatio-Temporal Databases**

## System Overview

Moving objects data are generated at a rapidly increasing scale, and natural language interfaces built on large language models let non-expert users access these data without learning a structured language. Existing interfaces follow the Text-to-SQL paradigm. However, SQL lacks native operators for moving objects, so the translated queries express spatial and temporal relationships in complex and error-prone ways, and the efficiency of the translated queries is delegated to the underlying optimizer. We therefore study the Text-to-Plan problem, which translates a natural language query directly into a query plan composed of moving objects operators and thereby unifies query translation with efficiency control. The problem is challenging because entity alignment, operator selection, and index usage induce an enormous candidate plan space, constructing a cost-optimal plan is NP-hard, and verifying imperative plans through execution is prohibitively expensive. We present NALPlan, to the best of our knowledge the first Text-to-Plan system for moving objects databases. NALPlan constrains generation with query plan models distilled from moving objects query semantics, broadens exploration through multi-path generation, prunes invalid candidates without execution through constraint-guided self-consistency checking, and selects efficient plans with few verification calls through cost-aware selection. Experiments show that NALPlan improves strict execution accuracy by 47.4\% relative to the best baseline and that cost-aware selection further improves correctness-gated plan efficiency by 28.6\%. The framework remains effective with small language models as the backbone.

The public entry point `run_pipeline.py` prints this stage order and the withheld-stage status. It is an interface demonstration, not a complete end-to-end executable release.

## Paper-to-Code Mapping

| Paper component | Public file | Main symbols |
|---|---|---|
| Sec. 5.1 query type and rule-chain preparation | `nalplan/main.py`, `nalplan/slm_postprocessor.py` | `QueryClassifier`, `ExecutableTemplateManager`, `SLMPostProcessor` |
| Sec. 5.1 operator retrieval | `nalplan/operator_retriever.py`, `nalplan/operator_retriever_embedding.py` | TF-IDF retrieval, embedding retrieval, weighted RRF |
| Sec. 5.2 multi-path candidate generation, Algorithm 1 | `nalplan/llm_generator.py` | `FewShotManager`, `LLMGenerator`, `MultiPathGenerator` |
| Sec. 6 executable and semantic pruning | `nalplan/constraint_pruner.py` | `Step1Validator`, `Step2Validator`, `Step3Validator`, `OperatorSemanticValidator`, `SemanticStep2Validator`, `ConflictPatternKB` |
| Sec. 7.1 index enumeration and cost-aware selection, Algorithm 2 | `nalplan/index_cost_optimizer.py` | `IndexKB`, `IndexEnumerator`, `DBAExperienceKB`, `LLMCostModel`, `QueryOptimizationPipeline` |
| Sec. 7.2 workload-aware semantic cache | `nalplan/workload_cache.py` | `QueryFeatureExtractor`, `WorkloadCache`, `CacheGuidedGenerator` |
| Candidate reranking and surface normalization | `nalplan/reranker.py`, `nalplan/postprocess.py` | `select_top1`, postprocessing helpers |
| SECONDO execution and evaluation metrics | `nalplan/secondo_connector.py`, `nalplan/eval_metrics.py` | connector, benchmark loading, EM/VER/EX/EFF utilities |
| Baseline adaptations | `nalplan/baselines.py` | DIN-SQL, DAIL-SQL, and OpenSearch-SQL runners |

The four contribution files listed for Sections 5.2, 6, 7.1, and 7.2 contain public class/function signatures and documentation, but their executable bodies are withheld.

## Repository Layout

```text
NALPlan-release/
|-- README.md
|-- .env.example
|-- requirements.txt
|-- run_pipeline.py
|-- nalplan/
|   |-- config.py
|   |-- main.py
|   |-- slm_postprocessor.py
|   |-- operator_retriever.py
|   |-- operator_retriever_embedding.py
|   |-- llm_generator.py              # interface-only
|   |-- constraint_pruner.py          # interface-only
|   |-- index_cost_optimizer.py       # interface-only
|   |-- workload_cache.py             # interface-only
|   |-- reranker.py
|   |-- postprocess.py
|   |-- secondo_connector.py
|   |-- eval_metrics.py
|   `-- baselines.py
|-- benchmark/
|   |-- nalplan_benchmark_examples.csv
|   `-- nalplan_benchmark_examples.json
`-- data/
    |-- README.md
    |-- berlintest/                   # public CSV/WKT data, including train trajectories
    `-- nanjingtest/                  # public geospatial CSV/WKT data only
```

## Installation

The intended environment is Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For full execution after the implementation is released, install SECONDO with the required algebras and restore the corresponding `berlintest` database objects, including `Trains`, `UnitTrains`, `UTOrdered`, and applicable spatial/temporal indexes. SECONDO is not bundled in this repository.

## Configuration

Configuration is centralized in `nalplan/config.py` and can be set through environment variables. The important variables are:

- `NALPLAN_LLM_BASE_URL`, `NALPLAN_LLM_API_KEY`, and `NALPLAN_LLM_MODEL`
- `NALPLAN_EMBEDDING_MODEL`
- `SECONDO_HOME`, `SECONDO_BIN`, `SECONDO_ALGEBRAS`, and `SECONDO_DATABASE_ROOT`
- `NALPLAN_NUM_PATHS=7`, `NALPLAN_CANDIDATES_PER_PATH=3`, and `NALPLAN_OPERATOR_DEPTH=30`
- `NALPLAN_TEMPERATURE=0.85` and `NALPLAN_QUERY_TIMEOUT=10`

No credential is stored in the repository. Set `NALPLAN_LLM_API_KEY` locally; never commit `.env` or API keys.

The original experiments use dataset-specific temporal anchors: `2003-11-20` for Berlin and `2024-06-15` for Nanjing. The Nanjing moving-object data are not part of this release.

## Data Release

`data/berlintest/` contains CSV/WKT exports of the public BerlinMOD-derived database, including `Trains.csv` and `Trains__Trip_units.csv`. `data/nanjingtest/` contains only public Nanjing geospatial layers such as districts, roads, universities, parks, and points of interest. Taxi trajectories and other moving-object relations for Nanjing are deliberately excluded because they are confidential.

The release keeps the base CSV/WKT representation and schema files. GeoJSON and exploded segment/vertex/sample-point encodings are omitted to control repository size. See `data/README.md` for format details.

## Benchmark Examples

`benchmark/nalplan_benchmark_examples.csv` and `.json` contain exactly 100 examples, 20 from each supported query family:

| Query family | Berlin | Nanjing | Total |
|---|---:|---:|---:|
| Time interval | 10 | 10 | 20 |
| Range | 10 | 10 | 20 |
| Nearest neighbor | 10 | 10 | 20 |
| Trajectory similarity | 10 | 10 | 20 |
| Join | 10 | 10 | 20 |
| **Total** | **50** | **50** | **100** |

Each record has `id`, `dataset`, `query_type`, `nlq`, and `gold_plan`. Nanjing examples preserve the paper's natural-language and plan context, but their gold plans reference moving-object relations that are not executable with the public Nanjing data. The complete 489-record benchmark and the 443 evaluable subset are withheld for the later full release.

## Withheld Artifacts

The following items are intentionally absent or reduced:

- Function bodies in `llm_generator.py`, `constraint_pruner.py`, `index_cost_optimizer.py`, and `workload_cache.py`.
- The 144 parameterized query-plan model table and few-shot example pool.
- Curated operator/conflict/DBA knowledge-base snapshots and embedding/TF-IDF caches.
- The complete benchmark, full experiment scripts, generated candidates, execution labels, and result tables.
- Nanjing moving-object data.

These omissions are part of the publication-release policy, not missing installation files.

## Running the Public Skeleton

From the repository root:

```bash
python run_pipeline.py --limit 1
```

The command loads a representative query and reports the intended NALPlan stage sequence. It will state that the contribution-stage implementations are withheld.

## Reproducibility and License

The paper reports experiments using Ubuntu, Python 3.10, SECONDO, and an LLM endpoint. Exact model, database, prompt, cache, and experiment details will accompany the complete artifact release after acceptance. The core implementation bodies and several learned/curated artifacts are withheld until paper acceptance. Interfaces, module organization, public data, and representative benchmark examples are provided so that the system architecture and experimental setting can be inspected without exposing the pending implementation.

No license has been selected for this staged release. Add the project license and the final citation metadata before publishing the GitHub repository.
