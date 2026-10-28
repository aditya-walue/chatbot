<script setup>
import { computed } from "vue";
import BrandMark from "./BrandMark.vue";
import { FeatherIcon } from "@/lib/ui";
import { suggestedQuestions } from "@/lib/suggestions";
import { useStore } from "@/store";
import { __ } from "@/lib/translate";

const { send, recentSessions, switchSession } = useStore();
const suggestions = suggestedQuestions();

const greeting = computed(() => {
	const hour = new Date().getHours();
	if (hour < 12) return __("Good morning");
	if (hour < 18) return __("Good afternoon");
	return __("Good evening");
});

// "Today" / "Yesterday" / weekday within a week / date after that.
function when(modified) {
	const date = new Date(String(modified).replace(" ", "T"));
	if (Number.isNaN(date.getTime())) return "";
	const days = Math.floor((new Date().setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / 864e5);
	if (days <= 0) return __("Today");
	if (days === 1) return __("Yesterday");
	if (days < 7) return date.toLocaleDateString(undefined, { weekday: "short" });
	return date.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
const recent = computed(() => (recentSessions.value || []).filter((s) => s.title).slice(0, 2));

defineProps({
	setup: { type: Boolean, default: false },
	hasModels: { type: Boolean, default: true },
	hasAgents: { type: Boolean, default: true },
});
</script>

<template>
	<div
		class="flex flex-1 flex-col pb-6 text-ink-gray-6"
		:class="setup ? 'items-center justify-center gap-2.5' : 'items-stretch justify-start'"
	>
		<BrandMark v-if="setup" :size="44" class="mb-2" />

		<template v-if="setup">
			<div class="text-lg font-semibold text-ink-gray-9">
				{{ __("Finish setup to start") }}
			</div>
			<div class="max-w-[260px] text-left text-xs leading-relaxed">
				{{ __("The assistant needs a few things configured first:") }}
			</div>
			<ol class="mt-1 flex max-w-[280px] flex-col gap-2 text-left">
				<li
					v-if="!hasModels"
					class="rounded-lg border border-outline-gray-1 bg-surface-gray-1 px-3 py-2 text-xs leading-relaxed text-ink-gray-8"
				>
					{{ __("Create and enable a Flow Model with your provider credentials.") }}
				</li>
				<li
					v-if="!hasAgents"
					class="rounded-lg border border-outline-gray-1 bg-surface-gray-1 px-3 py-2 text-xs leading-relaxed text-ink-gray-8"
				>
					{{
						__(
							"Enable a Flow Agent (Flow Agent is created automatically once a model exists)."
						)
					}}
				</li>
				<li
					class="rounded-lg border border-outline-gray-1 bg-surface-gray-1 px-3 py-2 text-xs leading-relaxed text-ink-gray-8"
				>
					{{
						__(
							"Enable Server Scripts in your site config so the assistant can run code."
						)
					}}
				</li>
			</ol>
		</template>

		<template v-else>
			<div class="flow-home">
				<div class="flow-eyebrow">{{ greeting }}</div>
				<h1 class="flow-headline">{{ __("What can I help you run today?") }}</h1>

				<div v-if="suggestions.length" class="flow-cards">
					<button
						v-for="s in suggestions"
						:key="s.text"
						class="flow-card"
						@click="send(s.text)"
					>
						<span class="flow-card-icon" :data-tone="s.tone">
							<FeatherIcon :name="s.icon" class="h-[18px] w-[18px]" />
						</span>
						<span class="flow-card-title">{{ s.title }}</span>
						<span class="flow-card-hint">{{ s.hint }}</span>
					</button>
				</div>

				<div v-if="recent.length" class="flow-recent">
					<div class="flow-recent-label">{{ __("Recent") }}</div>
					<button
						v-for="r in recent"
						:key="r.name"
						class="flow-recent-row"
						@click="switchSession(r.name)"
					>
						<FeatherIcon name="clock" class="h-4 w-4 shrink-0" />
						<span class="truncate">{{ r.title }}</span>
						<span class="flow-recent-when">{{ when(r.modified) }}</span>
					</button>
				</div>
			</div>
		</template>
	</div>
</template>

<style scoped>
.flow-home {
	display: flex;
	width: 100%;
	flex-direction: column;
	align-items: stretch;
	text-align: left;
}
.flow-eyebrow {
	font-size: 13px;
	font-weight: 600;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--flow-accent);
}
.flow-headline {
	margin: 6px 0 0;
	font-family: "Instrument Serif", Georgia, serif;
	font-style: normal;
	font-weight: 400;
	font-size: 32px;
	line-height: 1.05;
	letter-spacing: -0.01em;
	color: var(--ink-gray-9);
}
.flow-cards {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 10px;
	margin-top: 16px;
}
.flow-card {
	display: flex;
	flex-direction: column;
	align-items: flex-start;
	gap: 2px;
	padding: 12px;
	border: 1px solid var(--outline-gray-2);
	border-radius: 18px;
	background: var(--surface-white);
	text-align: left;
	box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
	transition: border-color 0.12s, box-shadow 0.12s, transform 0.12s;
}
.flow-card:hover {
	border-color: var(--flow-accent);
	box-shadow: 0 6px 18px -10px rgb(0 0 0 / 0.25);
	transform: translateY(-1px);
}
.flow-card-icon {
	display: inline-flex;
	height: 30px;
	width: 30px;
	align-items: center;
	justify-content: center;
	margin-bottom: 6px;
	border-radius: 9px;
}
.flow-card-icon[data-tone="green"] { background: #e3ecf8; color: #234b80; }
.flow-card-icon[data-tone="blue"] { background: #e8edf8; color: #2c3f86; }
.flow-card-icon[data-tone="amber"] { background: #f6ece0; color: #8a5a1c; }
.flow-card-icon[data-tone="violet"] { background: #ece9f7; color: #44388f; }
.flow-card-title {
	font-size: 14px;
	font-weight: 600;
	line-height: 1.25;
	color: var(--ink-gray-9);
}
.flow-card-hint {
	font-size: 12px;
	color: var(--ink-gray-5);
}
.flow-recent {
	margin-top: 16px;
}
.flow-recent-label {
	padding: 0 4px 6px;
	font-size: 13px;
	color: var(--ink-gray-5);
}
.flow-recent-row {
	display: flex;
	width: 100%;
	align-items: center;
	gap: 12px;
	padding: 7px 4px;
	border-radius: 10px;
	text-align: left;
	font-size: 14px;
	color: var(--ink-gray-8);
}
.flow-recent-row svg {
	color: var(--ink-gray-4);
}
.flow-recent-row:hover {
	background: var(--surface-gray-2);
}
.flow-recent-when {
	margin-left: auto;
	padding-left: 12px;
	font-size: 13px;
	white-space: nowrap;
	color: var(--ink-gray-5);
}
</style>
