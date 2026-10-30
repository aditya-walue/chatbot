<script setup>
import SessionsMenu from "./SessionsMenu.vue";
import { FeatherIcon } from "@/lib/ui";
import { useStore } from "@/store";
import { __ } from "@/lib/translate";

const props = defineProps({ onToggleFullscreen: { type: Function, default: null } });
const emit = defineEmits(["close"]);
const { recentSessions, switchSession, newChat, fullscreen, modelLoadingStatus } = useStore();
// The site's company (from the desk boot data), else the host, beside the "Online" status.
const site = frappe?.boot?.sysdefaults?.company || window.location.hostname;
</script>

<template>
	<header class="flow-header">
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
			<span v-else class="flex min-w-0 items-center gap-1.5 text-xs text-ink-gray-5">
				<span class="h-2 w-2 shrink-0 rounded-full bg-[#34c759]"></span>
				<span class="truncate">{{ __("Online") }} · {{ site }}</span>
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
