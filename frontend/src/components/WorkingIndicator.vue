<script setup>
import { __ } from "@/lib/translate";

defineProps({ label: { type: String, default: "" } });
</script>

<template>
	<!-- A status line (e.g. the browser model loading) when there is one, else three dots. -->
	<div v-if="label" class="flex items-center gap-2">
		<span class="flow-pulse-dot" aria-hidden="true"></span>
		<span class="flow-shimmer-text text-sm">{{ label }}</span>
	</div>
	<div v-else class="flow-typing" role="status" :aria-label="__('Working')">
		<span></span><span></span><span></span>
	</div>
</template>

<style scoped>
.flow-typing {
	display: inline-flex;
	align-items: center;
	gap: 5px;
	height: 20px;
}
.flow-typing span {
	height: 7px;
	width: 7px;
	border-radius: 999px;
	background: var(--ink-gray-4);
	animation: flow-typing 1.2s ease-in-out infinite;
}
.flow-typing span:nth-child(2) {
	animation-delay: 0.15s;
}
.flow-typing span:nth-child(3) {
	animation-delay: 0.3s;
}
@keyframes flow-typing {
	0%, 60%, 100% {
		opacity: 0.3;
		transform: translateY(0);
	}
	30% {
		opacity: 1;
		transform: translateY(-3px);
	}
}
</style>
