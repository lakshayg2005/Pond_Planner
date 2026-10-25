import { withSpan } from "../telemetry";

const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

function describeError(data, status) {
  const detail = data?.detail;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        const where = Array.isArray(item?.loc) ? item.loc.slice(1).join(" → ") : "";
        return where ? `${where}: ${item.msg}` : item?.msg ?? JSON.stringify(item);
      })
      .join("; ");
  }

  if (status === 502 || status === 503 || status === 504) {
    return "The server is busy or unavailable. Please try again in a moment.";
  }

  return `Request failed (HTTP ${status})`;
}

async function request(path, options) {
  let response;

  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new Error(
      "Could not reach the analysis server. Check your connection and try again."
    );
  }

  let data = null;
  try {
    data = await response.json();
  } catch {
    // non-JSON body
  }

  if (!response.ok) throw new Error(describeError(data, response.status));
  if (!data || data.status !== "success") {
    throw new Error(data?.message || "Analysis failed.");
  }

  return data;
}

/** Analyse a drawn area. `ring` is a closed array of [lng, lat]. */
export function analyzeArea(ring, params, signal) {
  const body = {
    polygon: ring,
    max_slope_percent: params.maxSlope,
    max_candidates: params.maxPonds,
  };
  if (params.rainfall) body.rainfall_mm = params.rainfall;
  if (params.runoff) body.runoff_coefficient = params.runoff;

  return withSpan(
    "ui.analyze_area",
    { "pond.vertices": ring.length - 1, "pond.max_candidates": params.maxPonds },
    () =>
      request("/api/analyze-area", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      })
  );
}

/** Analyse an uploaded KML/KMZ contour map. */
export function analyzeContourFile(file, params, signal) {
  const query = new URLSearchParams({
    max_slope_percent: String(params.maxSlope),
    max_candidates: String(params.maxPonds),
  });
  if (params.rainfall) query.set("rainfall_mm", String(params.rainfall));
  if (params.runoff) query.set("runoff_coefficient", String(params.runoff));

  const form = new FormData();
  form.append("contour_map", file);

  return withSpan(
    "ui.analyze_contour_file",
    { "pond.upload.filename": file.name, "pond.upload.bytes": file.size },
    () =>
      request(`/api/catchment/analyze?${query}`, {
        method: "POST",
        body: form,
        signal,
      })
  );
}

export async function fetchServerInfo() {
  try {
    const response = await fetch(`${API_URL}/api/health`);
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

export async function searchPlace(query) {
  const url =
    "https://nominatim.openstreetmap.org/search?format=json&limit=5&q=" +
    encodeURIComponent(query);
  const response = await fetch(url);
  if (!response.ok) throw new Error("Search failed");
  const items = await response.json();

  return items.map((item) => ({
    name: item.display_name,
    bounds: [
      [Number(item.boundingbox[0]), Number(item.boundingbox[2])],
      [Number(item.boundingbox[1]), Number(item.boundingbox[3])],
    ],
  }));
}
