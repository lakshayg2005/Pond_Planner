// OpenTelemetry in the browser: page-load and fetch spans, plus one span per
// analysis the user starts. Every same-origin API call carries a W3C
// `traceparent` header, so the backend's spans join the same trace.
//
// Spans are sent to the backend (/otel/v1/traces), which forwards them to the
// collector: the browser cannot reach the collector directly.

import { context, trace, SpanStatusCode } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { DocumentLoadInstrumentation } from "@opentelemetry/instrumentation-document-load";
import { FetchInstrumentation } from "@opentelemetry/instrumentation-fetch";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { BatchSpanProcessor, WebTracerProvider } from "@opentelemetry/sdk-trace-web";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");
const EXPORT_URL = `${API_URL}/otel/v1/traces`;

const provider = new WebTracerProvider({
  resource: resourceFromAttributes({
    [ATTR_SERVICE_NAME]: "pond-planner-web",
    "browser.language": navigator.language,
  }),
  spanProcessors: [
    new BatchSpanProcessor(new OTLPTraceExporter({ url: EXPORT_URL }), {
      scheduledDelayMillis: 2000,
    }),
  ],
});
provider.register();

registerInstrumentations({
  instrumentations: [
    new DocumentLoadInstrumentation(),
    new FetchInstrumentation({
      // Never trace the exporter's own requests.
      ignoreUrls: [/\/otel\/v1\/traces/],
      // Same-origin calls get the header automatically; the dev server
      // talks to the API on another port, so allow that too.
      propagateTraceHeaderCorsUrls: API_URL ? [new RegExp(`^${API_URL}`)] : [],
    }),
  ],
});

const tracer = trace.getTracer("pond-planner-web");

/**
 * Run `work` inside a span named `name`. The fetch started inside it becomes
 * a child, and so does everything the backend does for that request.
 */
export async function withSpan(name, attributes, work) {
  const span = tracer.startSpan(name, { attributes });
  try {
    const result = await context.with(trace.setSpan(context.active(), span), work);
    span.setStatus({ code: SpanStatusCode.OK });
    return result;
  } catch (error) {
    span.recordException(error);
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
    throw error;
  } finally {
    span.end();
  }
}
