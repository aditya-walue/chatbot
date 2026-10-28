<script setup>
import { ref, computed, watch, nextTick } from "vue";
import AttachmentChip from "./AttachmentChip.vue";
import { FeatherIcon } from "@/lib/ui";
import { useStore } from "@/store";
import { __ } from "@/lib/translate";

const {
	selectedAgent,
	attachments,
	sending,
	paused,
	locked,
	loaded,
	needsSetup,
	uploading,
	focusTick,
	agentLabel,
	send,
	stopRun,
	attachFiles,
	removeAttachment,
} = useStore();

// Backend (flow/boot.py) is the single source of truth for supported file types.
const ACCEPT = computed(() =>
	(frappe.boot.flow_supported_file_types || []).map((ext) => `.${ext}`).join(",")
);

const text = ref("");
const el = ref(null);
const fileInput = ref(null);
const dragging = ref(false);
const submitting = ref(false);

const inputDisabled = computed(
	() => !loaded.value || sending.value || paused.value || needsSetup.value || submitting.value
);

const canSend = computed(() => text.value.trim() && !inputDisabled.value && !submitting.value && !uploading.value);
const placeholder = computed(() => {
	if (!loaded.value) return __("Loading…");
	if (needsSetup.value) return __("Setup required…");
	return __("Ask {0}…", [agentLabel(selectedAgent.value)]);
});

async function submit() {
	if (!canSend.value || submitting.value || sending.value) return;
	submitting.value = true;
	const content = text.value.trim();
	text.value = "";
	nextTick(resize);
	try {
		await send(content);
	} finally {
		submitting.value = false;
	}
}

function pickFiles() {
	fileInput.value?.click();
}

function onFilesPicked(e) {
	attachFiles(e.target.files);
	e.target.value = ""; // allow re-picking the same file
}

function onDragOver(e) {
	if (inputDisabled.value) return;
	e.preventDefault();
	dragging.value = true;
}

function onDragLeave(e) {
	if (e.currentTarget.contains(e.relatedTarget)) return;
	dragging.value = false;
}

function onDrop(e) {
	e.preventDefault();
	dragging.value = false;
	if (inputDisabled.value) return;
	attachFiles(e.dataTransfer.files);
}

function onKeydown(e) {
	if (e.key === "Enter" && !e.shiftKey) {
		e.preventDefault();
		submit();
	}
}

function resize() {
	const t = el.value;
	if (!t) return;
	t.style.height = "auto";
	t.style.height = `${Math.min(t.scrollHeight, 160)}px`;
}

watch(focusTick, () => nextTick(() => el.value?.focus()));
</script>

<template>
	<div
		class="flow-composer absolute inset-x-4 bottom-4 mx-auto flex max-w-3xl flex-col gap-1 px-3 pb-2.5 pt-2"
	>
		<div v-if="attachments.length" class="flex flex-wrap gap-2">
			<AttachmentChip
				v-for="a in attachments"
				:key="a.uid"
				:file-name="a.file_name"
				:file-size="a.file_size"
				:status="a.status"
				:error="a.error"
				removable
				@remove="removeAttachment(a.uid)"
			/>
		</div>

		<textarea
			ref="el"
			v-model="text"
			rows="1"
			:placeholder="placeholder"
			:disabled="inputDisabled"
			class="max-h-40 min-h-[44px] w-full resize-none border-0 bg-transparent px-0.5 pt-1 text-base font-normal leading-relaxed text-ink-gray-9 outline-none placeholder:text-ink-gray-4"
			@keydown="onKeydown"
			@input="resize"
		></textarea>

		<div class="flex items-center gap-2">
			<span class="flex-1"></span>

			<button
				v-if="sending"
				class="flow-send is-stop"
				:title="__('Stop')"
				:aria-label="__('Stop')"
				@click="stopRun"
			>
				<span class="h-2.5 w-2.5 rounded-[2px] bg-current"></span>
			</button>
			<button
				v-else
				class="flow-send"
				:disabled="!canSend"
				:title="__('Send')"
				:aria-label="__('Send')"
				@click="submit"
			>
				<FeatherIcon name="arrow-up" :stroke-width="2.4" class="h-4 w-4" />
			</button>
		</div>
	</div>
</template>
