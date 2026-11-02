<script setup>
import SessionsMenu from "./SessionsMenu.vue";
import { FeatherIcon } from "@/lib/ui";
import { useStore } from "@/store";
import { __ } from "@/lib/translate";

const props = defineProps({ onToggleFullscreen: { type: Function, default: null } });
const emit = defineEmits(["close"]);
const { recentSessions, switchSession, newChat, fullscreen, modelLoadingStatus } = useStore();
</script>

<template>
	<header class="flow-header">
		<span class="flow-header-tile" aria-hidden="true">
			<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
				<path d="M6.5 3.5h11A3.5 3.5 0 0 1 21 7v6.5a3.5 3.5 0 0 1-3.5 3.5H11l-4.6 3.6a.6.6 0 0 1-.97-.5V17A3.5 3.5 0 0 1 3 13.5V7a3.5 3.5 0 0 1 3.5-3.5z" />
			</svg>
		</span>
		<div class="ml-1 flex min-w-0 flex-col leading-tight">
			<span class="text-[17px] font-semibold tracking-tight text-ink-gray-9">{{
				__("Flow")
			}}</span>
			<span
				v-if="modelLoadingStatus"
				class="flow-shimmer-text max-w-[220px] truncate text-xs text-ink-gray-5"
				:title="modelLoadingStatus"
			>
				{{ modelLoadingStatus }}
			</span>
			<span v-else class="flow-online">
				<span class="flow-online-dot"></span>
				<span>{{ __("Online") }}</span>
			</span>
		</div>
		<span class="flex-1"></span>

		<SessionsMenu :sessions="recentSessions" @select="switchSession" />

		<button class="flow-icon-btn" :title="__('New chat')" @click="newChat">
			<FeatherIcon name="plus" :stroke-width="2" class="h-4 w-4" />
		</button>
		<button
			class="flow-icon-btn"
			:title="fullscreen ? __('Exit full screen') : __('Full screen')"
			@click="props.onToggleFullscreen && props.onToggleFullscreen()"
		>
			<FeatherIcon
				:name="fullscreen ? 'minimize' : 'maximize'"
				:stroke-width="2"
				class="h-4 w-4"
			/>
		</button>
		<button class="flow-icon-btn" :title="__('Close (Ctrl+I)')" @click="emit('close')">
			<FeatherIcon name="x" :stroke-width="2" class="h-4 w-4" />
		</button>
	</header>
</template>
