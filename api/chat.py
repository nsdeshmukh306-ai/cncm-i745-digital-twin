import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

PARAM_EXTRACTOR_SYSTEM = """You are a parameter extractor for a
Saccharomyces boulardii CNCM I-745 digital twin simulator.
Extract simulation parameters from natural language queries.
Always return ONLY valid JSON with exactly these keys:
{
  "glucose": float between -0.1 and -20.0,
  "oxygen": float between 0.0 and -20.0,
  "nh4": float between -0.1 and -5.0,
  "pi": float between -0.1 and -2.0,
  "simulation_type": one of: fba, surrogate, host, regulatory, genome,
  "gut_zone": one of: stomach, duodenum, ileum, colon, none,
  "intent_summary": one sentence describing what user wants to simulate
}
Biological mappings for gut zones:
- stomach: glucose=-0.5, oxygen=0.0, pH=2.0 (anaerobic, acidic)
- duodenum: glucose=-1.0, oxygen=-2.0, pH=6.0 (microaerobic)
- ileum: glucose=-1.5, oxygen=-5.0, pH=7.0 (aerobic)
- colon: glucose=-0.5, oxygen=0.0, pH=7.2 (anaerobic)
Defaults: glucose=-1.65, oxygen=-2.0, nh4=-1.0, pi=-0.5
Return ONLY the JSON, no other text."""

SCIENTIFIC_WRITER_SYSTEM = """You are a scientific writer specializing
in probiotic microbiology and computational systems biology.
You have deep knowledge of Saccharomyces boulardii CNCM I-745,
genome-scale metabolic modeling, flux balance analysis, and
gut microbiome interactions.

Given simulation results from a CNCM I-745 digital twin, write a
rigorous scientifically accurate interpretation.

Return ONLY valid JSON with these exact keys:
{
  "plain_english": "2-3 sentence plain language summary",
  "scientific_summary": "3-4 sentence technical summary with exact units,
    reference to computational method (FBA/CNN/ODE), and quantitative context",
  "biological_context": "2-3 sentences on broader probiotic biology,
    referencing known CNCM I-745 mechanisms from literature",
  "clinical_relevance": "1-2 sentences on potential clinical implications
    for gut health or disease management",
  "confidence": "high, medium, or low based on: high=FBA feasible+normal range,
    medium=edge of training distribution, low=infeasible or extrapolated",
  "suggested_followup": "one specific follow-up simulation to suggest",
  "key_finding": "one sentence maximum — the single most important result"
}
Return ONLY the JSON, no other text."""


def extract_simulation_params(user_message: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=PARAM_EXTRACTOR_SYSTEM,
        messages=[{"role": "user", "content": user_message}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_scientific_response(simulation_results: dict,
                                  user_message: str, params: dict) -> dict:
    context = f"""User query: {user_message}

Simulation parameters used:
{json.dumps(params, indent=2)}

Raw simulation results:
{json.dumps(simulation_results, indent=2)}

Validated CNCM I-745 reference values:
- Baseline growth rate: 0.0858 h⁻¹ (glucose-limited)
- Optimal temperature: 37°C
- NF-kB suppression in literature: 60-75% (Thomas et al. 2019)
- Toxin neutralization (CAMP factor): 70-80% at 120 min

Please interpret these results in the context of CNCM I-745 probiotic biology."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=SCIENTIFIC_WRITER_SYSTEM,
        messages=[{"role": "user", "content": context}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
