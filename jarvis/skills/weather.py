"""Weather skill — Open-Meteo (no API key required)."""
from __future__ import annotations

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> (label, icon)
WMO: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "\U0001f324️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "\U0001f32b️"), 48: ("Rime fog", "\U0001f32b️"),
    51: ("Light drizzle", "\U0001f326️"), 53: ("Drizzle", "\U0001f326️"),
    55: ("Dense drizzle", "\U0001f326️"),
    56: ("Freezing drizzle", "\U0001f327️"), 57: ("Freezing drizzle", "\U0001f327️"),
    61: ("Light rain", "\U0001f326️"), 63: ("Rain", "\U0001f327️"),
    65: ("Heavy rain", "\U0001f327️"),
    66: ("Freezing rain", "\U0001f327️"), 67: ("Freezing rain", "\U0001f327️"),
    71: ("Light snow", "\U0001f328️"), 73: ("Snow", "\U0001f328️"),
    75: ("Heavy snow", "❄️"), 77: ("Snow grains", "\U0001f328️"),
    80: ("Light showers", "\U0001f326️"), 81: ("Showers", "\U0001f327️"),
    82: ("Violent showers", "⛈️"),
    85: ("Snow showers", "\U0001f328️"), 86: ("Snow showers", "\U0001f328️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm w/ hail", "⛈️"),
    99: ("Thunderstorm w/ hail", "⛈️"),
}


async def _geocode(client: httpx.AsyncClient, city: str) -> dict | None:
    r = await client.get(GEOCODE_URL, params={"name": city, "count": 1})
    r.raise_for_status()
    results = r.json().get("results")
    return results[0] if results else None


async def get_weather(city: str, units: str = "metric") -> dict:
    """Return current conditions + today's high/low, shaped for the HUD."""
    temp_unit = "celsius" if units == "metric" else "fahrenheit"
    wind_unit = "kmh" if units == "metric" else "mph"
    deg = "°C" if units == "metric" else "°F"
    async with httpx.AsyncClient(timeout=15) as client:
        place = await _geocode(client, city)
        if not place:
            return {"ok": False, "error": f"Could not find location '{city}'."}
        params = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
        }
        r = await client.get(FORECAST_URL, params=params)
        r.raise_for_status()
        data = r.json()

    cur = data.get("current", {})
    daily = data.get("daily", {})
    code = int(cur.get("weather_code", 0))
    label, icon = WMO.get(code, ("Unknown", "❓"))

    def first(key: str):
        v = daily.get(key)
        return v[0] if isinstance(v, list) and v else None

    high, low = first("temperature_2m_max"), first("temperature_2m_min")
    return {
        "ok": True,
        "city": place.get("name", city),
        "country": place.get("country", ""),
        "icon": icon,
        "condition": label,
        "temp": round(cur.get("temperature_2m", 0)),
        "feels_like": round(cur.get("apparent_temperature", 0)),
        "humidity": cur.get("relative_humidity_2m"),
        "wind": round(cur.get("wind_speed_10m", 0)),
        "wind_unit": wind_unit,
        "high": round(high) if high is not None else None,
        "low": round(low) if low is not None else None,
        "precip_chance": first("precipitation_probability_max"),
        "unit": deg,
    }
