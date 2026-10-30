<script setup>
import { computed } from "vue";
import MarkdownText from "./MarkdownText.vue";
import ActivityGroup from "./ActivityGroup.vue";
import ConfirmCard from "./ConfirmCard.vue";
import WorkingIndicator from "./WorkingIndicator.vue";
import { toolError } from "@/lib/toolMeta";
import { useStore } from "@/store";

const props = defineProps({ message: { type: Object, required: true } });
const { answerQuestion, toolApproval } = useStore();

const questionByKey = computed(() => new Map(props.message.questions.map((q) => [q.key, q])));

// A confirmation tool (per the agent's tool map), or one that has/had a question.
function isApproval(part) {
	const q = questionByKey.value.get(part.id);
	return toolApproval.value[part.name] === true || q !== undefined || part.approval !== null;
}

// Group parts for render: text → prose, approval tool → own line (card if pending),
// other tools → merged activity group. Rendering-only.
const items = computed(() => {
	const out = [];
	for (const part of props.message.parts) {
		if (part.type !== "tool") {
			out.push({ kind: "text", id: part.id, part });
			continue;
		}
		// Steps behind the answer ("Finding relevant DocTypes", "Replying") are not shown; a failed
		// one stays, since with no answer text after it the failure is all the user sees.
		if (!isApproval(part) && !toolError(part.result)) continue;
		if (isApproval(part)) {
			const q = questionByKey.value.get(part.id);
			if (q && q._answer === undefined)
				out.push({ kind: "confirm", id: part.id, question: q, part });
			else out.push({ kind: "approval", id: part.id, parts: [part] });
			continue;
		}
		const last = out[out.length - 1];
		if (last?.kind === "activity") last.parts.push(part);
		else out.push({ kind: "activity", id: part.id, parts: [part] });
	}
	return out;
});

// The working dots show until the answer text starts to arrive: the steps themselves are hidden.
const showWorking = computed(
	() => props.message.pending && !items.value.some((i) => i.kind === "text" && i.part.text)
);
</script>

<template>
	<div class="flex items-start gap-3">
		<span class="flow-avatar mt-1" aria-hidden="true">
			<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
				<path d="M6.5 3.5h11A3.5 3.5 0 0 1 21 7v6.5a3.5 3.5 0 0 1-3.5 3.5H11l-4.6 3.6a.6.6 0 0 1-.97-.5V17A3.5 3.5 0 0 1 3 13.5V7a3.5 3.5 0 0 1 3.5-3.5z" />
			</svg>
		</span>
		<div class="flow-parts flex min-w-0 flex-1 flex-col">
			<template v-for="(item, i) in items" :key="item.id">
				<div v-if="item.kind === 'text'" class="flow-bubble">
					<MarkdownText :part="item.part" />
				</div>
				<ConfirmCard
					v-else-if="item.kind === 'confirm'"
					:question="item.question"
					:tool="item.part"
					@answer="(answer) => answerQuestion(message, item.question, answer)"
				/>
				<ActivityGroup
					v-else
					:parts="item.parts"
					:sealed="i < items.length - 1"
					:live="message.pending"
				/>
			</template>

			<WorkingIndicator v-if="message.status" :label="message.status" />
			<WorkingIndicator v-else-if="showWorking" />
		</div>
	</div>
</template>
