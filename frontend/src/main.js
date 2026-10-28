import { createApp, watch } from "vue";
import App from "@/App.vue";
import { useStore } from "@/store";
import * as api from "@/api/client";
import { readPanelState, writePanelState } from "@/lib/panelState";
import "@/index.css";

// Popup window: anchored bottom-right, never full height. "Full screen" is the expanded view.
const POPUP_WIDTH = 404;
const POPUP_HEIGHT = 640;
const MIN_WIDTH = 340;
const MIN_HEIGHT = 420;
const EDGE = 24; // gap between the popup and the window edges
const LAYOUT_VERSION = 2; // older saved states were a docked full-height sidebar
const EASE = "cubic-bezier(0.2, 0.8, 0.2, 1)";

const BLUE = "#2f5f9e";
const SPARKLE = `<path d="M12 7.2l1.15 3.1 3.1 1.15-3.1 1.15L12 15.7l-1.15-3.1-3.1-1.15 3.1-1.15L12 7.2z" fill="currentColor" stroke="none" />`;
// Speech bubble with a sparkle inside. The launcher sits outside #flow-root, so no Vue/CSS here.
const LAUNCHER_ICON = `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
	<path d="M12 3.2a8.3 8.3 0 1 1-3.9 15.6L3.6 20l1.2-4.3A8.3 8.3 0 0 1 12 3.2z" />
	${SPARKLE}
</svg>`;
const BRAND_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" aria-hidden="true">
	<path d="M12 3.5l2 6.5 6.5 2-6.5 2-2 6.5-2-6.5-6.5-2 6.5-2 2-6.5z" />
</svg>`;
const CLOSE_ICON = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>`;
const ARROW_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>`;

// Popup chat window injected into the Frappe desk. The Vue app (with real
// frappe-ui components) mounts inside #flow-root; all bundle CSS is scoped to
// that id so nothing leaks onto the desk.
class FlowPanel {
	constructor() {
		const saved = readPanelState();
		this.visible = Boolean(saved.open);
		const current = saved.v === LAYOUT_VERSION;
		this._width = (current && saved.width) || POPUP_WIDTH;
		this._height = (current && saved.height) || POPUP_HEIGHT;
		// The popup is the default; a saved full-screen preference wins on reload.
		this._initialFullscreen = current ? Boolean(saved.fullscreen) : false;

		this._mount();
		this._addLauncher();
		this._syncTheme();
		this._registerShortcut();

		watch(this.store.sessionName, () => this._persist());
	}

	get fullscreen() {
		return this.store.fullscreen.value;
	}

	_mount() {
		this.store = useStore();
		this.store.fullscreen.value = this._initialFullscreen;

		this.root = document.createElement("div");
		this.root.id = "flow-root";
		Object.assign(this.root.style, {
			position: "fixed",
			zIndex: "1040",
			overflow: "hidden",
			transformOrigin: "bottom right",
		});
		this._applyLayout(false);
		document.body.appendChild(this.root);
		window.addEventListener("resize", () => this._applyLayout(false));

		this.app = createApp(App, {
			onClose: () => this.hide(),
			onToggleFullscreen: () => this.toggleFullscreen(),
		});
		this.app.mount(this.root);

		this._addResizeHandle();
	}

	// Where the window sits: a bottom-right popup sized to the viewport, or the whole viewport.
	// Sizes are clamped so a saved size never overflows a smaller window.
	_applyLayout(animate = true) {
		const style = this.root.style;
		const full = this.fullscreen;
		const maxWidth = window.innerWidth - EDGE * 2;
		const maxHeight = window.innerHeight - EDGE * 2;
		const width = Math.min(maxWidth, Math.max(MIN_WIDTH, this._width));
		const height = Math.min(maxHeight, Math.max(MIN_HEIGHT, this._height));
		const geometry = `width 0.26s ${EASE}, height 0.26s ${EASE}, right 0.26s ${EASE}, bottom 0.26s ${EASE}, border-radius 0.26s ${EASE}`;
		const visibility = `opacity 0.18s ease, transform 0.24s ${EASE}, visibility 0s linear ${this.visible ? "0s" : "0.24s"}`;
		style.transition = animate ? `${geometry}, ${visibility}` : visibility;
		style.right = full ? "0" : `${EDGE}px`;
		style.bottom = full ? "0" : `${EDGE}px`;
		style.width = full ? "100vw" : `${width}px`;
		style.height = full ? "100vh" : `${height}px`;
		style.borderRadius = full ? "0" : "18px";
		style.border = full ? "none" : "1px solid rgba(128, 128, 128, 0.22)";
		style.boxShadow = full
			? "none"
			: "0 24px 64px -16px rgba(16, 24, 40, 0.32), 0 4px 14px rgba(16, 24, 40, 0.08)";
		// Hidden: sunk toward the launcher, not clickable.
		style.opacity = this.visible ? "1" : "0";
		style.transform = this.visible ? "none" : "translateY(14px) scale(0.96)";
		style.visibility = this.visible ? "visible" : "hidden";
		style.pointerEvents = this.visible ? "auto" : "none";
	}

	// Grab area on the popup's top-left corner: dragging resizes it from there (the popup stays
	// anchored bottom-right). Appended after mount so Vue's render doesn't clobber it.
	_addResizeHandle() {
		const handle = document.createElement("div");
		Object.assign(handle.style, {
			position: "absolute",
			top: "0",
			left: "0",
			width: "18px",
			height: "18px",
			cursor: "nwse-resize",
			zIndex: "10",
		});
		this.root.appendChild(handle);

		const onMove = (e) => {
			this._width = Math.min(window.innerWidth - EDGE * 2, Math.max(MIN_WIDTH, window.innerWidth - EDGE - e.clientX));
			this._height = Math.min(window.innerHeight - EDGE * 2, Math.max(MIN_HEIGHT, window.innerHeight - EDGE - e.clientY));
			this._applyLayout(false);
		};
		const onUp = () => {
			document.removeEventListener("mousemove", onMove);
			document.removeEventListener("mouseup", onUp);
			document.body.style.userSelect = "";
			this._persist();
		};
		handle.addEventListener("mousedown", (e) => {
			if (this.fullscreen) return;
			e.preventDefault();
			document.body.style.userSelect = "none";
			document.addEventListener("mousemove", onMove);
			document.addEventListener("mouseup", onUp);
		});
	}

	// Chat-bubble button in the bottom-right corner of every desk page. Hidden while the panel
	// is open.
	_addLauncher() {
		const button = document.createElement("button");
		button.type = "button";
		button.className = "flow-launcher";
		button.title = __("Open Flow (Ctrl+I)");
		button.setAttribute("aria-label", __("Open Flow"));
		button.innerHTML = LAUNCHER_ICON;
		// Small green "online" dot on the top-right edge.
		const dot = document.createElement("span");
		Object.assign(dot.style, {
			position: "absolute",
			top: "1px",
			right: "1px",
			width: "11px",
			height: "11px",
			borderRadius: "50%",
			background: "#4caf7a",
			boxShadow: "0 0 0 2px #ffffff",
		});
		button.appendChild(dot);
		Object.assign(button.style, {
			position: "fixed",
			right: "24px",
			bottom: "24px",
			width: "56px",
			height: "56px",
			borderRadius: "50%",
			border: "none",
			display: "flex",
			alignItems: "center",
			justifyContent: "center",
			background: BLUE,
			color: "#ffffff",
			boxShadow: "0 0 0 14px rgba(47, 95, 158, 0.12), 0 10px 24px -6px rgba(47, 95, 158, 0.5)",
			cursor: "pointer",
			zIndex: "1039",
			transition: "transform 0.15s ease",
		});
		button.addEventListener("mouseenter", () => (button.style.transform = "scale(1.06)"));
		button.addEventListener("mouseleave", () => (button.style.transform = ""));
		button.addEventListener("click", () => this.show());
		document.body.appendChild(button);
		this.launcher = button;
		this._syncLauncher();
		this._peek();
	}

	// A mini chat card over the launcher on every desk page load, picking up the user's most
	// recent chat: their last question and Flow's reply (or typing dots if it never finished),
	// with a reply box and an Open link. Nothing is shown for a user with no earlier chat.
	async _peek() {
		let chat;
		try {
			chat = await this._lastChat();
		} catch (e) {
			console.debug("[flow] peek card: could not load the last chat", e);
			return;
		}
		if (!chat || this.visible) {
			console.debug("[flow] peek card skipped:", chat ? "panel is open" : "no earlier chat");
			return;
		}

		const el = (tag, css, text) => {
			const node = document.createElement(tag);
			Object.assign(node.style, css);
			if (text) node.textContent = text;
			return node;
		};
		const card = el("div", {
			position: "fixed",
			right: "24px",
			bottom: "96px",
			width: "290px",
			maxWidth: "calc(100vw - 48px)",
			background: "#ffffff",
			color: "#1f2933",
			borderRadius: "24px",
			boxShadow: "0 18px 48px -12px rgba(0,0,0,0.3), 0 0 0 1px rgba(0,0,0,0.05)",
			font: "400 14px/1.45 system-ui, -apple-system, sans-serif",
			zIndex: "1039",
			opacity: "0",
			transform: "translateY(8px)",
			transition: "opacity 0.25s ease, transform 0.25s ease",
			});

		const head = el("div", { display: "flex", alignItems: "center", gap: "10px", padding: "14px 16px 6px" });
		const tile = el("span", { display: "inline-flex", width: "36px", height: "36px", alignItems: "center", justifyContent: "center", borderRadius: "10px", background: BLUE, color: "#fff", flex: "none" });
		tile.innerHTML = BRAND_ICON;
		const titles = el("div", { flex: "1", minWidth: "0" });
		titles.append(
			el("div", { fontWeight: "600", fontSize: "15px" }, __("Flow")),
			el("div", { fontSize: "12.5px", color: "#6b7280" }, chat.reply ? __("From your recent question") : __("Picking up where you left off"))
		);
		const closeBtn = el("button", { border: "none", background: "none", color: "#4b5563", cursor: "pointer", padding: "4px", display: "inline-flex" });
		closeBtn.innerHTML = CLOSE_ICON;
		closeBtn.type = "button";
		closeBtn.setAttribute("aria-label", __("Close"));
		head.append(tile, titles, closeBtn);

		const body = el("div", { padding: "6px 16px 12px", display: "flex", flexDirection: "column", gap: "10px" });
		body.append(
			el("div", { alignSelf: "flex-end", maxWidth: "85%", padding: "8px 14px", borderRadius: "16px", background: "#e8eef8", wordBreak: "break-word" }, chat.question)
		);
		if (chat.reply) {
			body.append(el("div", { alignSelf: "flex-start", padding: "10px 14px", borderRadius: "16px", border: "1px solid #e5e7eb", wordBreak: "break-word" }, chat.reply));
		} else {
			const dots = el("div", { alignSelf: "flex-start", display: "inline-flex", gap: "5px", padding: "10px 14px", borderRadius: "16px", background: "#f1f3f2" });
			for (let i = 0; i < 3; i++) dots.append(el("span", { width: "7px", height: "7px", borderRadius: "50%", background: i === 2 ? "#6b7280" : "#c4c9c7" }));
			body.append(dots);
		}

		const foot = el("div", { display: "flex", alignItems: "center", gap: "10px", padding: "13px 16px", borderTop: "1px solid #eceeed", background: "#fbfbfa", borderRadius: "0 0 24px 24px" });
		const input = el("input", { flex: "1", minWidth: "0", border: "none", outline: "none", background: "transparent", font: "inherit", color: "inherit" });
		input.placeholder = __("Reply to Flow…");
		const open = el("button", { border: "none", background: "none", font: "600 14px system-ui, sans-serif", color: BLUE, cursor: "pointer", whiteSpace: "nowrap", display: "inline-flex", alignItems: "center", gap: "6px" });
		open.append(el("span", {}, __("Open")));
		open.insertAdjacentHTML("beforeend", ARROW_ICON);
		open.type = "button";
		foot.append(input, open);
		card.append(head, body, foot);
		// A small tail pointing down at the launcher.
		card.append(el("span", { position: "absolute", right: "32px", bottom: "-6px", width: "14px", height: "14px", background: "#fbfbfa", transform: "rotate(45deg)", borderRadius: "0 0 3px 0", boxShadow: "2px 2px 3px -1px rgba(0,0,0,0.08)" }));

		const close = () => {
			clearTimeout(this._peekTimer);
			card.style.opacity = "0";
			setTimeout(() => card.remove(), 300);
		};
		const resume = async (text) => {
			close();
			this._setVisible(true);
			await this.store.switchSession(chat.session);
			if (text) this.store.send(text);
			this._persist();
		};
		closeBtn.addEventListener("click", close);
		open.addEventListener("click", () => resume());
		input.addEventListener("keydown", (e) => {
			if (e.key === "Enter" && input.value.trim()) resume(input.value.trim());
		});
		// Stay while the user is typing or hovering; otherwise tuck away after a while.
		const arm = () => (this._peekTimer = setTimeout(close, 3000));
		card.addEventListener("mouseenter", () => clearTimeout(this._peekTimer));
		card.addEventListener("mouseleave", arm);
		input.addEventListener("input", () => clearTimeout(this._peekTimer));
		this.launcher.addEventListener("click", close);
		this.peekBubble = card;
		document.body.appendChild(card);
		setTimeout(() => {
			card.style.opacity = "1";
			card.style.transform = "";
			arm();
		}, 1500);
	}

	// The user's most recent chat: its last question and the reply after it, if any.
	async _lastChat() {
		const [latest] = await api.loadHistory();
		if (!latest) return null;
		const doc = await api.getSession(latest.name);
		const rows = doc.messages || [];
		const asked = rows.map((m) => m.role).lastIndexOf("user");
		if (asked < 0) return null;
		const answer = rows.slice(asked + 1).reverse().find((m) => m.role === "assistant" && m.content);
		const plain = (text) => text.replace(/[*_`#>\[\]]|\(\/[^)]*\)/g, "").replace(/\s+/g, " ").trim();
		const clip = (text, n) => (text.length > n ? text.slice(0, n - 1) + "…" : text);
		return {
			session: latest.name,
			question: clip(plain(rows[asked].content || ""), 90),
			reply: answer ? clip(plain(answer.content), 160) : "",
		};
	}

	_syncLauncher() {
		this.launcher.hidden = this.visible;
		if (this.visible && this.peekBubble) this.peekBubble.remove();
	}

	// Mirror the desk's light/dark theme onto the panel root so scoped tokens
	// resolve to the right palette.
	_syncTheme() {
		const apply = () => {
			const theme = document.documentElement.getAttribute("data-theme") || "light";
			this.root.setAttribute("data-theme", theme);
		};
		apply();
		new MutationObserver(apply).observe(document.documentElement, {
			attributes: true,
			attributeFilter: ["data-theme"],
		});
	}

	_registerShortcut() {
		frappe.ui.keys.add_shortcut({
			shortcut: "ctrl+i",
			action: () => this.toggle(),
			description: __("Toggle Flow panel"),
			ignore_inputs: true,
		});
	}

	show() {
		this._setVisible(true);
		this.store.restoreSession();
		this._persist();
	}

	hide() {
		this._setVisible(false);
		this._persist();
	}

	_setVisible(visible) {
		this.visible = visible;
		this._applyLayout();
		this._syncLauncher();
	}

	toggle() {
		this.visible ? this.hide() : this.show();
	}

	// Expand the popup to the full viewport, or shrink it back. State lives in the store so the
	// header icon tracks it reactively.
	toggleFullscreen() {
		this.store.fullscreen.value = !this.fullscreen;
		this._applyLayout();
		this._persist();
	}

	_persist() {
		writePanelState({
			open: this.visible,
			v: LAYOUT_VERSION,
			fullscreen: this.fullscreen,
			width: this._width,
			height: this._height,
			session: this.store.sessionName.value,
		});
	}
}

frappe.provide("frappe.flow");
$(document).on("app_ready", () => {
	frappe.flow.panel = new FlowPanel();
});
