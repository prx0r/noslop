"""Weather worldpack: evolve optimal response format for WEATHER_CHECK/WEATHER_FORECAST.

World: query open-meteo → format response with template
Candidate: format template (the gene being evolved)
Quality gate: word-overlap score ≥ 0.80 against ground truth
"""
from __future__ import annotations

import json
import random
import urllib.request
import urllib.parse
from dataclasses import dataclass, field

from cogym_kernel.kernel.contracts import (
    ActionResult, ActionSpec, Metric, MetricVector, WorldSpec
)


# --- Ground truth templates (what the validator expects) ---

WEATHER_CHECK_TRUTHS = [
    "Currently {temp_c}C and {condition} in {city}.",
    "It is {temp_c}C and {condition} in {city}.",
    "The weather in {city} is {temp_c}C and {condition}.",
    "{city}: {temp_c}C, {condition}.",
    "Right now in {city} it is {temp_c}C and {condition}.",
]

WEATHER_FORECAST_TRUTHS = [
    "The forecast for {city}: {temp_c}C and {condition}.",
    "{city} will be {temp_c}C and {condition}.",
    "Expect {temp_c}C and {condition} in {city}.",
]

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "rime fog", 51: "light drizzle", 53: "moderate drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}

# Cities for generating test instances
CITIES = [
    ("London", 51.51, -0.13), ("Paris", 48.86, 2.35), ("Tokyo", 35.68, 139.69),
    ("New York", 40.71, -74.01), ("Berlin", 52.52, 13.41), ("Sydney", -33.87, 151.21),
    ("Dubai", 25.20, 55.27), ("Singapore", 1.35, 103.82), ("Mumbai", 19.08, 72.88),
    ("Cairo", 30.04, 31.24), ("Moscow", 55.76, 37.62), ("Rio de Janeiro", -22.91, -43.17),
    ("Toronto", 43.65, -79.38), ("Seoul", 37.57, 126.98), ("Bangkok", 13.76, 100.50),
]


@dataclass
class WeatherState:
    seed: int
    city_name: str
    city_lat: float
    city_lon: float
    temp_c: float
    condition: str
    query_type: str  # "check" or "forecast"
    response: str | None = None
    score: float | None = None


class WeatherWorld:
    def __init__(self, query_type: str = "check"):
        self.query_type = query_type
        self._spec = None

    @property
    def world_spec(self) -> WorldSpec:
        if self._spec is None:
            self._spec = WorldSpec(
                world_kind=f"weather.{self.query_type}",
                version="1",
                instance_set_hash="cities-v1",
                environment_hash="open-meteo-v1",
                oracle_hash="wmo-codes-v1",
            )
        return self._spec

    @property
    def worldpack_id(self) -> str:
        from cogym_kernel.kernel.ids import content_id
        return content_id("wp", {"kind": f"weather.{self.query_type}", "v": 1})

    def _fetch_weather(self, lat: float, lon: float) -> tuple[float, str]:
        """Fetch current weather from open-meteo."""
        try:
            params = urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,weather_code",
                "timezone": "auto",
            })
            url = f"https://api.open-meteo.com/v1/forecast?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "noslop-miner/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            temp = data["current"]["temperature_2m"]
            code = data["current"]["weather_code"]
            condition = WMO_CODES.get(code, f"weather code {code}")
            return temp, condition
        except Exception:
            # Fallback: use seed-based deterministic fake data
            rng = random.Random(hash((lat, lon)))
            temp = round(rng.uniform(-5, 35), 1)
            condition = rng.choice(list(WMO_CODES.values()))
            return temp, condition

    def reset(self, *, instance_id: str, seed: int) -> WeatherState:
        rng = random.Random(seed)
        city = CITIES[seed % len(CITIES)]
        temp, condition = self._fetch_weather(city[1], city[2])
        # Deterministic ground truth template (same seed = same template)
        truths = WEATHER_CHECK_TRUTHS if self.query_type == "check" else WEATHER_FORECAST_TRUTHS
        gt_template = truths[seed % len(truths)]
        ground_truth = gt_template.format(
            temp_c=temp, condition=condition, city=city[0]
        )
        state = WeatherState(
            seed=seed, city_name=city[0], city_lat=city[1], city_lon=city[2],
            temp_c=temp, condition=condition, query_type=self.query_type,
        )
        # Store ground truth on state for scoring
        state._ground_truth = ground_truth
        return state

    def observe(self, state: WeatherState) -> dict:
        return {
            "city": state.city_name,
            "temp_c": state.temp_c,
            "condition": state.condition,
            "query_type": state.query_type,
            "response": state.response,
            "score": state.score,
        }

    def actions(self, state: WeatherState) -> tuple[ActionSpec, ...]:
        if state.response is None:
            return (
                ActionSpec(
                    kind="FORMAT_RESPONSE",
                    payload={
                        "city": state.city_name,
                        "temp_c": state.temp_c,
                        "condition": state.condition,
                        "query_type": state.query_type,
                    },
                    executor_kind="deterministic",
                ),
            )
        return ()

    def apply(self, state: WeatherState, action: ActionSpec,
              result: ActionResult) -> WeatherState:
        if action.kind == "FORMAT_RESPONSE" and result.status == "ok":
            response = result.payload.get("response", "")
            new_state = WeatherState(
                seed=state.seed, city_name=state.city_name,
                city_lat=state.city_lat, city_lon=state.city_lon,
                temp_c=state.temp_c, condition=state.condition,
                query_type=state.query_type, response=response,
            )
            new_state._ground_truth = getattr(state, "_ground_truth", None)
            return new_state
        return state

    def terminal(self, state: WeatherState) -> bool:
        return state.response is not None

    def score(self, state: WeatherState) -> MetricVector:
        if state.response is None:
            return MetricVector(metrics=(Metric("overlap", 0.0, "max"),))

        ground_truth = getattr(state, "_ground_truth", None)
        if ground_truth is None:
            return MetricVector(metrics=(Metric("overlap", 0.0, "max"),))

        overlap = _word_overlap(state.response, ground_truth)
        length_penalty = max(0, 1.0 - abs(len(state.response) - len(ground_truth)) / 100)

        return MetricVector(metrics=(
            Metric("overlap", round(overlap, 4), "max"),
            Metric("length_fit", round(length_penalty, 4), "max"),
        ))


def _word_overlap(a: str, b: str) -> float:
    """Precision-based word overlap (simulates the WASM scorer)."""
    words_a = a.lower().split()
    words_b = b.lower().split()
    if not words_a or not words_b:
        return 0.0
    # Count how many words from a appear in b
    b_set = set(words_b)
    matches = sum(1 for w in words_a if w in b_set)
    return matches / len(words_a) if words_a else 0.0
