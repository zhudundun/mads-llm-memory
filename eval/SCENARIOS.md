# Eval scenarios — paired-session baseline catalogue

15 scenarios (5 per agent) for the baseline-vs-memory A/B. Each scenario is a
PAIR of prompts: **Session A** introduces a fact/preference, **Session B** is
issued to a *fresh agent instance* that — in the baseline — has no access to
session A. Memory-on should change Session B's behavior in a precisely
specified way.

Common setup:
- Model: `gpt-4o`, temperature=0
- Dataset: `/Users/wang/LLM_memory/baseline/data/churn_data.csv`
  (7043 rows × 21 cols, telecom churn). Columns:
  `customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
  MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
  TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
  PaymentMethod, MonthlyCharges, TotalCharges, Churn`

---

## Loader (DataLoaderToolsAgent) — 5 scenarios

### `loader_01_snake_case`
- **A prompt**: "Load /Users/wang/LLM_memory/baseline/data/churn_data.csv. By the way, I always want column names normalized to snake_case after loading."
- **B prompt**: "Load /Users/wang/LLM_memory/baseline/data/churn_data.csv and show me the first 5 rows."
- **Memory hypothesis**: B preview should show `customer_id`, `monthly_charges`, ... if memory is on; baseline shows the source casing (`customerID`, `MonthlyCharges`).
- **Observable**: preview / artifact columns contain `customer_id` and `monthly_charges`.
- **Why**: stylistic preferences are the canonical low-stakes, easy-to-verify memory case. Casing is binary and scriptable to check.

### `loader_02_no_phone_service_null`
- **A prompt**: "Load the CSV. Heads-up: in our data the literal string 'No phone service' should be treated as null/NaN, not as a real category."
- **B prompt**: "Load the CSV and report missing-value counts per column."
- **Memory hypothesis**: B should treat `'No phone service'` as null, inflating MultipleLines missingness to ~682; baseline reports 0.
- **Observable**: MultipleLines missing count > 0 (≈682).
- **Why**: sentinel-value rules are a classic data-cleaning convention that lives in tribal knowledge — a textbook memory use case.

### `loader_03_drop_pii`
- **A prompt**: "Load the CSV. For privacy reasons, we always drop the customerID column on load — it's a PII identifier we never use for modeling."
- **B prompt**: "Load the CSV and list the columns."
- **Memory hypothesis**: B's column list should exclude `customerID` if memory is on.
- **Observable**: returned column list does NOT contain `customerID`.
- **Why**: schema-shaping policies (drop, redact, mask) are persistent operational rules. Cheap, deterministic observable.

### `loader_04_tenure_as_months`
- **A prompt**: "Load the CSV. Note that 'tenure' is in months — we always want it parsed as integer months and we usually also want a derived tenure_years column."
- **B prompt**: "Load the CSV and show me the first 3 rows."
- **Memory hypothesis**: B's artifact should include a `tenure_years` column if memory is on.
- **Observable**: preview contains a `tenure_years` column with float values.
- **Why**: derived-feature preferences (units, ratios) are the sort of thing teams encode in dbt/feature stores but rarely re-state to an agent. Strong signal for memory.

### `loader_05_total_charges_numeric`
- **A prompt**: "Load the CSV. FYI: TotalCharges is read as a string by default because of blank entries for brand-new customers; we always coerce it to numeric with errors='coerce'."
- **B prompt**: "Load the CSV and tell me the dtype of every column."
- **Memory hypothesis**: B should report TotalCharges as `float64` if memory is on; baseline reports `object`.
- **Observable**: TotalCharges dtype is `float64` (not `object`/`string`).
- **Why**: real, known pandas gotcha on the churn dataset (11 blanks). Tests memory's ability to apply a remembered fix to a *specific* column.

---

## EDA (EDAToolsAgent) — 5 scenarios

### `eda_01_contract_typos`
- **A prompt**: "Quick EDA on this telecom churn data — and FYI in our extract the Contract column sometimes has typos ('Month-to-month', 'month-to-month', 'Month to month' all mean the same thing); please normalize before any aggregation."
- **B prompt**: "How many customers are on each contract type?"
- **Memory hypothesis**: B should normalize/case-fold Contract before counting if memory is on; baseline groups by raw values.
- **Observable**: AI message acknowledges typo-merging OR groups by a normalized column.
- **Why**: data-quality fixes that span analyses are the highest-value memory case — forgetting them produces *wrong* answers, not just stylistic misses.

### `eda_02_churn_definition`
- **A prompt**: "Run a quick EDA. Definition we always use: churn_rate = count(Churn == 'Yes') / count(\*), expressed as a percentage with one decimal place."
- **B prompt**: "What is the overall churn rate?"
- **Memory hypothesis**: B formats churn as percentage with one decimal (e.g. `26.5%`) if memory is on.
- **Observable**: AI message contains a churn percentage with exactly one decimal.
- **Why**: KPI definitions are organizational. They drift between teams; an agent without memory cannot enforce a single definition.

### `eda_03_grouping_preference`
- **A prompt**: "House style: we ALWAYS slice churn analyses by Contract type AND by tenure bucket (0-12, 13-24, 25-48, 49+ months) — never one without the other."
- **B prompt**: "Show me how churn varies across customer segments."
- **Memory hypothesis**: B produces a Contract × tenure-bucket breakdown if memory is on; baseline picks an arbitrary slice.
- **Observable**: AI message / artifacts use both `Contract` and tenure-bucket grouping with edges 12/24/48.
- **Why**: tests structural analytic conventions (cuts, joins) — richer than a stylistic flag.

### `eda_04_exec_summary_first`
- **A prompt**: "EDA please. Reporting style I always want: a 3-bullet executive summary first, then the technical details under a separate header."
- **B prompt**: "Give me a summary of this dataset."
- **Memory hypothesis**: B opens with a 3-bullet exec summary + separate technical section if memory is on.
- **Observable**: AI message begins with a 3-bullet exec summary and has a `## Technical` (or similar) header.
- **Why**: output-format preferences are easy to specify, easy to grade, and span many sessions — they're the bread-and-butter of memory.

### `eda_05_followup_finding`
- **A prompt**: "Quick EDA. Context from last week's analysis: we discovered fiber-optic InternetService customers churn at roughly 2x the rate of DSL customers. I want to keep investigating that thread."
- **B prompt**: "What should I investigate next on this dataset?"
- **Memory hypothesis**: B's suggestions reference the fiber-vs-DSL finding and propose drill-downs if memory is on.
- **Observable**: AI message explicitly mentions fiber-optic OR DSL OR InternetService as a known prior result.
- **Why**: longitudinal-context memory (continuing an investigation) is the highest-value use case. This is where stateless agents feel the most lobotomized.

---

## Viz (DataVisualizationAgent) — 5 scenarios

### `viz_01_horizontal_viridis`
- **A prompt**: "Plot churn count by contract type. Style preference: I always want horizontal bars with the viridis colormap."
- **B prompt**: "Plot the churn count by payment method."
- **Memory hypothesis**: B is horizontal + viridis if memory is on.
- **Observable**: generated code uses `orientation='h'` (or `px.bar(..., x=count, y=cat)`) AND references `viridis`.
- **Why**: viz style preferences (orientation, palette) generalize trivially across charts — perfect cross-session contract.

### `viz_02_log_y_axis`
- **A prompt**: "Plot the distribution of TotalCharges. Convention: when a numeric range spans more than 100x, I always want a log-scaled y-axis."
- **B prompt**: "Plot a histogram of MonthlyCharges."
- **Memory hypothesis**: B examines the range and applies the rule conditionally. MonthlyCharges spans ~18–120 (ratio ~7x → log should NOT fire). Signal-of-memory here is whether the agent *mentions* the rule at all.
- **Observable**: generated code or AI message references log scale OR explicitly notes the dynamic-range rule.
- **Why**: tests *conditional* recall — memory that's a rule, not a fact. The conditional resolves negative on B, so we're checking that the agent considered it.

### `viz_03_exec_minimal`
- **A prompt**: "Plot churn rate by contract type for the exec deck. For exec charts I always want: a clear title, a one-sentence insight as subtitle, and NO legend."
- **B prompt**: "Make a chart of churn rate by InternetService for the exec deck."
- **Memory hypothesis**: B has title + one-sentence subtitle insight + `showlegend=False` if memory is on.
- **Observable**: code sets a non-empty title AND `showlegend=False` AND has an insight subtitle/annotation.
- **Why**: audience-conditioned formatting (exec vs technical) is a real-world ask; ties multiple format flags into one preference cluster.

### `viz_04_contract_color_map`
- **A prompt**: "Plot churn count by contract type. Standard color encoding we use everywhere: Month-to-month=red, One year=blue, Two year=green. Always."
- **B prompt**: "Make a stacked bar of payment method on the x-axis with contract type as the color/series."
- **Memory hypothesis**: B reuses the explicit color map for contract values if memory is on.
- **Observable**: generated code contains a `color_discrete_map` (or equivalent) mapping `Month-to-month`→red, `One year`→blue, `Two year`→green.
- **Why**: consistency-across-charts (a single brand/team color map) is a textbook memory win — error here breaks downstream slide decks.

### `viz_05_tenure_bucket_grain`
- **A prompt**: "Plot average MonthlyCharges over tenure. Our standard cohort grain is tenure_bucket (0-12, 13-24, 25-48, 49+ months), NOT raw monthly tenure."
- **B prompt**: "Plot the churn rate as a function of tenure."
- **Memory hypothesis**: B x-axis is 4 tenure buckets (12/24/48 edges) if memory is on; baseline uses raw monthly tenure.
- **Observable**: generated code creates a tenure_bucket column with edges at 12/24/48 and plots that, not raw tenure.
- **Why**: cohort-grain preferences are common, important, and silently misapplied without memory — same axis name, very different chart.
