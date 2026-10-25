// All network calls live here.

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    });
  } catch {
    throw new ApiError(
      "Cannot reach the API. Is the Django server running?",
      0,
      null,
    );
  }

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => ({}))
    : {};

  if (!response.ok) {
    const message =
      payload?.error?.message ||
      (response.status === 429
        ? "Too many requests. Wait a moment and try again."
        : `Request failed (${response.status}).`);
    throw new ApiError(message, response.status, payload);
  }
  return payload;
}

function queryString(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  });
  const result = search.toString();
  return result ? `?${result}` : "";
}

export const api = {
  health: () => request("/health/"),
  stats: () => request("/stats/"),
  modelInfo: () => request("/model-info/"),
  predict: (text) =>
    request("/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),
  predictBatch: (file, textColumn = "") => {
    const form = new FormData();
    form.append("file", file);
    if (textColumn.trim()) form.append("text_column", textColumn.trim());
    return request("/predict/batch/", { method: "POST", body: form });
  },
  history: ({ page = 1, pageSize = 8, label = "", source = "", search = "" } = {}) =>
    request(
      `/history/${queryString({
        page,
        page_size: pageSize,
        label,
        source,
        search,
      })}`,
    ),
};

export { API_BASE };
