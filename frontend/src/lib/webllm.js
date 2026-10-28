import { __ } from "@/lib/translate";

// Runs browser-model calls (Flow Models with a `webllm/<model>` id) on WebGPU via
// @mlc-ai/web-llm. The server's agent loop sends each call over the run stream as an
// `llm_request` event and waits for the reply we post back (see flow/lib/webllm.py).
//
// The library is several MB, so it's loaded from the CDN on first use instead of being
// bundled into the panel that every desk page loads.
const WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm@0.2.85";
// WebLLM's default (4k) is too small for an agent's instructions plus tool schemas.
// Keep in sync with DEFAULT_CONTEXT_WINDOW in flow/lib/webllm.py.
const DEFAULT_CONTEXT_WINDOW = 16384;
const TOOL_CALL_TAG = "<tool_call>";
const STOP_SEQUENCES = ["</tool_call>", "<tool_response"];

let libPromise = null;
let engine = null;
let engineKey = null;
let engineInitPromise = null;
let deviceLostHandlerAttached = false;

function loadLib() {
	libPromise ||= import(/* @vite-ignore */ WEBLLM_URL).catch((e) => {
		libPromise = null;
		throw e;
	});
	return libPromise;
}

// Reset the engine instance (e.g. after unrecoverable error or GPU device loss)
export function resetEngine() {
	if (engine) {
		try {
			engine.unload?.();
		} catch {
			// ignore unload errors
		}
	}
	engine = null;
	engineKey = null;
	engineInitPromise = null;
	deviceLostHandlerAttached = false;
}

export function isEngineLoaded(model, contextWindow = DEFAULT_CONTEXT_WINDOW) {
	const key = `${model}:${contextWindow}`;
	return Boolean(engine && engineKey === key);
}

// Verify WebGPU availability and adapter compatibility
export async function verifyWebGPU() {
	if (typeof navigator === "undefined" || !navigator.gpu) {
		throw new Error(
			__("This browser does not support WebGPU, which browser models require. Use a recent Chrome or Edge with hardware acceleration enabled.")
		);
	}
	try {
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			throw new Error(
				__("WebGPU is available in your browser, but no compatible graphics adapter was found. Check your browser GPU flags or hardware acceleration.")
			);
		}
		return adapter;
	} catch (e) {
		if (e.message?.includes("WebGPU")) throw e;
		throw new Error(__("Failed to initialize WebGPU: {0}", [e.message || String(e)]));
	}
}

function setupDeviceLostHandler(adapter) {
	if (deviceLostHandlerAttached || !adapter) return;
	deviceLostHandlerAttached = true;
	adapter.requestDevice?.().then((device) => {
		device?.lost?.then((info) => {
			console.warn("Flow: WebGPU device lost:", info?.message, info?.reason);
			resetEngine();
		});
	}).catch(() => {});
}

// Listen to tab visibility changes to detect background suspension
if (typeof document !== "undefined") {
	document.addEventListener("visibilitychange", () => {
		if (document.visibilityState === "visible" && engine) {
			// Ensure engine is responsive after tab wake
			if (typeof navigator !== "undefined" && !navigator.gpu) {
				resetEngine();
			}
		}
	});
}

// One engine per active application session; switching models reloads it in place.
// Concurrent calls reuse the in-flight initialization promise.
export async function getEngine(model, contextWindow = DEFAULT_CONTEXT_WINDOW, onStatus) {
	const key = `${model}:${contextWindow}`;
	if (engine && engineKey === key) return engine;
	if (engineInitPromise && engineKey === key) return engineInitPromise;

	// Shown at once: the first question waits on the model's download and start-up.
	if (onStatus) onStatus(__("Model is loading, please wait…"));
	await verifyWebGPU();

	engineKey = key;
	engineInitPromise = (async () => {
		let webllm;
		try {
			webllm = await loadLib();
		} catch (e) {
			throw new Error(__("Failed to load WebLLM from CDN: {0}", [e.message || String(e)]));
		}

		// WebLLM reports progress per phase (fetching, caching, compiling), so the raw value jumps
		// back down between phases; show the furthest point reached instead.
		let furthest = 0;
		const initProgressCallback = (p) => {
			if (onStatus) {
				furthest = Math.max(furthest, Math.round((p.progress || 0) * 100));
				onStatus(__("Model is loading, please wait… {0}% (the first time takes a few minutes)", [furthest]), p);
			}
		};
		const chatOpts = { context_window_size: contextWindow };

		try {
			if (engine) {
				await engine.reload(model, chatOpts);
			} else {
				engine = await webllm.CreateMLCEngine(model, { initProgressCallback }, chatOpts);
			}
			if (navigator.gpu) {
				navigator.gpu.requestAdapter().then(setupDeviceLostHandler).catch(() => {});
			}
			if (onStatus) onStatus("");
			return engine;
		} catch (err) {
			resetEngine();
			const msg = err?.message || String(err);
			if (msg.includes("out of memory") || msg.includes("OutOfMemory") || msg.includes("allocation")) {
				throw new Error(__("Insufficient GPU memory to run {0}. Try closing other browser tabs or using a smaller model.", [model]));
			}
			if (msg.includes("quota") || msg.includes("QuotaExceeded")) {
				throw new Error(__("Storage quota exceeded while caching model weights. Please clear browser storage."));
			}
			if (msg.includes("fetch") || msg.includes("network") || msg.includes("NetworkError")) {
				throw new Error(__("Failed to download model weights. Please check your internet connection."));
			}
			throw new Error(__("Model initialization failed: {0}", [msg]));
		} finally {
			engineInitPromise = null;
		}
	})();

	return engineInitPromise;
}

// Fetch and load a model ahead of its first call (e.g. on session start or greeting),
// so questions don't wait on the multi-GB download.
export async function preloadModel({ model, context_window_size }, onStatus) {
	if (!navigator.gpu || !model) return;
	try {
		await getEngine(model, context_window_size || DEFAULT_CONTEXT_WINDOW, onStatus);
	} catch (e) {
		console.warn("Flow: could not preload the model", e);
	} finally {
		if (onStatus) onStatus("");
	}
}

// Text the user should see: everything before the first tool call, minus a trailing
// partial "<tool_call>" tag that may still be streaming in.
function visibleText(full) {
	const cut = full.indexOf(TOOL_CALL_TAG);
	if (cut !== -1) return full.slice(0, cut);
	for (let n = Math.min(TOOL_CALL_TAG.length - 1, full.length); n > 0; n--) {
		if (TOOL_CALL_TAG.startsWith(full.slice(-n))) return full.slice(0, -n);
	}
	return full;
}

// Strict mode: the reply is one JSON object. While it streams, show only the text of its
// "reply" field (decoding JSON string escapes); a {"tool": ...} object shows nothing.
function strictVisibleText(full) {
	const match = full.match(/^\s*\{\s*"reply"\s*:\s*"/);
	if (!match) return "";
	let out = "";
	for (let i = match[0].length; i < full.length; i++) {
		const ch = full[i];
		if (ch === '"') break;
		if (ch !== "\\") {
			out += ch;
			continue;
		}
		const next = full[i + 1];
		if (next === undefined) break; // escape still streaming in
		if (next === "u") {
			const hex = full.slice(i + 2, i + 6);
			if (hex.length < 4) break;
			out += String.fromCharCode(parseInt(hex, 16));
			i += 5;
			continue;
		}
		out += { n: "\n", t: "\t", r: "", b: "", f: "" }[next] ?? next;
		i += 1;
	}
	return out;
}

// Run one model call. Streams visible text through `onText` (or falls back to non-streaming)
// and returns { content, usage } with the raw reply for the server.
export async function runBrowserRequest(request, { onText, onStatus, signal }) {
	const { context_window_size, ...params } = request.params || {};
	const eng = await getEngine(
		request.model,
		context_window_size || DEFAULT_CONTEXT_WINDOW,
		onStatus
	);
	if (signal?.aborted) throw new DOMException("Aborted", "AbortError");

	const onAbort = () => eng.interruptGenerate();
	signal?.addEventListener("abort", onAbort);
	const base = { ...params, messages: request.messages, stream: true, stream_options: { include_usage: true } };
	let strict = Boolean(request.response_schema);
	let isStreaming = true;

	try {
		let chunks;
		try {
			chunks = await eng.chat.completions.create(
				strict
					? // Decoding is constrained to the schema: valid JSON with tool call or reply.
					  { ...base, response_format: { type: "json_object", schema: request.response_schema } }
					: // End the turn once a tool call is written.
					  { stop: STOP_SEQUENCES, ...base }
			);
		} catch (e) {
			if (signal?.aborted) throw e;
			if (strict) {
				console.warn("Flow: strict JSON streaming unavailable, falling back to non-strict", e);
				strict = false;
				try {
					chunks = await eng.chat.completions.create({ stop: STOP_SEQUENCES, ...base });
				} catch (e2) {
					if (signal?.aborted) throw e2;
					console.warn("Flow: streaming failed, falling back to non-streaming", e2);
					isStreaming = false;
					chunks = await eng.chat.completions.create({
						...params,
						messages: request.messages,
						stream: false,
						stop: STOP_SEQUENCES,
					});
				}
			} else {
				console.warn("Flow: streaming failed, falling back to non-streaming", e);
				isStreaming = false;
				chunks = await eng.chat.completions.create({
					...params,
					messages: request.messages,
					stream: false,
					stop: STOP_SEQUENCES,
				});
			}
		}

		if (!isStreaming) {
			const text = chunks?.choices?.[0]?.message?.content || "";
			if (onText && text) onText(text);
			return { content: text, usage: chunks?.usage || {} };
		}

		const visibleOf = strict ? strictVisibleText : visibleText;
		let full = "";
		let shown = 0;
		let usage = {};
		for await (const chunk of chunks) {
			if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
			const delta = chunk.choices?.[0]?.delta?.content;
			if (delta) {
				full += delta;
				const visible = visibleOf(full);
				if (visible.length > shown) {
					onText(visible.slice(shown));
					shown = visible.length;
				}
			}
			if (chunk.usage) usage = chunk.usage;
		}
		if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
		return { content: full, usage };
	} finally {
		signal?.removeEventListener("abort", onAbort);
	}
}
