import os
import json
from openai import OpenAI

# Construct with a placeholder when the key is absent so that *import never
# fails* (the API would otherwise crash on startup if DEEPSEEK_API_KEY were
# unset). Real calls use the env key in production; with the placeholder they
# fail gracefully and are caught by the /chat fallback handler.
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY") or "sk-deepseek-key-not-set",
    base_url="https://api.deepseek.com",
)

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
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=500,
        messages=[
            {"role": "system", "content": PARAM_EXTRACTOR_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def build_scientific_context(growth_rate=None, gut_zone="none", surrogate_r2=None,
                             barrier_score=None, nfkb_suppression_pct=None) -> str:
    """Render the live-simulation-state block injected into the system prompt.

    Grounds the assistant in the current digital-twin state so multi-turn
    answers stay consistent with the most recent simulation.
    """
    return (
        "LIVE SIMULATION STATE (ground all answers in these values):\n"
        f"- Current growth rate: {growth_rate if growth_rate is not None else 'N/A'} h⁻¹\n"
        f"- Active gut zone: {gut_zone}\n"
        f"- CNN surrogate cross-validated R²: {surrogate_r2 if surrogate_r2 is not None else 'N/A'}\n"
        f"- Epithelial barrier-integrity score: {barrier_score if barrier_score is not None else 'N/A'}\n"
        f"- NF-κB suppression vs control: {nfkb_suppression_pct if nfkb_suppression_pct is not None else 'N/A'}%\n"
    )


def generate_scientific_response(simulation_results: dict,
                                  user_message: str, params: dict,
                                  history: list | None = None,
                                  scientific_context: str | None = None) -> dict:
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

    system_prompt = SCIENTIFIC_WRITER_SYSTEM
    if scientific_context:
        system_prompt = SCIENTIFIC_WRITER_SYSTEM + "\n\n" + scientific_context

    messages = [{"role": "system", "content": system_prompt}]
    # Prior multi-turn dialogue, if supplied ({role, content} dicts).
    for turn in (history or []):
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": context})

    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1200,
        messages=messages,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


EXPLAIN_FLUX_SYSTEM = """You are a metabolic-modelling expert explaining a single
reaction from the Saccharomyces boulardii CNCM I-745 genome-scale metabolic model
to a scientifically literate but non-specialist reader.
Given a reaction's ID, name, current flux value and whether its flux changed under
the active gut condition, write a concise plain-English explanation (3-5 sentences)
covering: (1) the reaction's biological role/pathway, (2) what the current flux
value means (direction and magnitude, units mmol gDW⁻¹ hr⁻¹), and (3) the
significance of any change under the gut condition. Be accurate; do not invent
numbers. Return plain text, no JSON, no markdown headers."""


def explain_reaction(reaction_id: str, reaction_name: str, flux_value: float,
                     changed: bool, gut_zone: str, baseline_flux: float) -> str:
    """Return a plain-English explanation of a reaction's biological role/flux."""
    prompt = (
        f"Reaction ID: {reaction_id}\n"
        f"Reaction name: {reaction_name}\n"
        f"Current flux (active condition '{gut_zone}'): {flux_value:.6f} mmol gDW⁻¹ hr⁻¹\n"
        f"Baseline flux (glucose-limited): {baseline_flux:.6f} mmol gDW⁻¹ hr⁻¹\n"
        f"Flux changed under gut condition: {'yes' if changed else 'no'}\n"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=400,
        messages=[
            {"role": "system", "content": EXPLAIN_FLUX_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()
